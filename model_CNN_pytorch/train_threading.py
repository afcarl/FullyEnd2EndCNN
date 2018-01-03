import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable

from tqdm import tqdm
from .models import SubCNN
from . import training_utils as tu
from ..utils import logging_utils as lu

from time import time
from . import batch_utils
import shutil
import os
import json


def train(settings):
    """Train NN for eye tracking regression

    Args:
        settings: StereoTrackerSettings instance
        iteration: Iteration of the regression cascade
    """

    try:
        shutil.rmtree("figures_pytorch")
    except Exception:
        pass

    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

    # Sampling grid for landmarks
    max_x, max_y = 240, 240
    patch_size = 32
    settings.patch_size = patch_size
    settings.max_x = max_x
    settings.max_y = max_y
    patch_shape = (patch_size, patch_size)
    patch_shape = np.array(patch_shape)
    patch_half_shape = np.require(np.round(patch_shape / 2), dtype=int)
    start = -patch_half_shape
    end = patch_half_shape
    sampling_grid = np.mgrid[start[0]:end[0], start[1]:end[1]]
    sampling_grid = sampling_grid.swapaxes(0, 2).swapaxes(0, 1).astype(np.float32)

    sg_size = sampling_grid.shape
    pts_size = (21, 2)

    sampling_grid = torch.FloatTensor(sampling_grid).cuda()
    sampling_grid = sampling_grid.contiguous().view((1,) + sg_size).expand((pts_size[0],) + sg_size)
    old_size = sampling_grid.size()
    sampling_grid = sampling_grid.contiguous().view((1,) + old_size)

    settings.sampling_grid = sampling_grid

    ###########################
    # Data and normalization
    ###########################

    # Get data
    val_img0, val_pts0, val_init0 = tu.load_data("data/processed/CNN_database.h5", "validation")
    tr_img0, tr_pts0, tr_init0 = tu.load_data("data/processed/CNN_database.h5", "training", size=val_img0.shape[0])

    # Get normalizations (stored in settings)
    tu.load_normalizations(settings,"data/processed/CNN_database.h5")

    # Random idx to plot figures.
    idxs_train = np.random.permutation(tr_img0.shape[0])[:4]
    idxs_val = np.random.permutation(val_img0.shape[0])[:4]

    ###########################
    # Neural Net
    ###########################

    n_filters = 64
    hidden_dim = 512
    input_dim = 21
    output_dim = 42

    list_models = [SubCNN(input_dim, output_dim, n_filters, hidden_dim).cuda() for i in range(settings.max_iteration)]

    for model in list_models:
        print(model)

    loss_fn = torch.nn.MSELoss(size_average=True).cuda()
    # loss_fn2 = torch.nn.L1Loss(size_average=True).cuda()

    ###########################
    # Monitoring
    ###########################

    # Initialize a dict to hold monitoring metrics
    d_losses = {"train_losses": [],
                "train_mean_shape_rmse": [],
                "train_std_shape_rmse": [],
                "val_mean_shape_rmse": [],
                "val_std_shape_rmse": [],
                "val_mean_shape_rmse_closed": [],
                "val_std_shape_rmse_closed": [],
                "val_mean_shape_rmse_open": [],
                "val_std_shape_rmse_open": [],
                "best_val_mean_shape_rmse": [],
                "duration": []}

    num_elem = tr_img0.shape[0]
    num_batches = num_elem // settings.batch_size
    list_batches = np.array_split(np.arange(num_elem), num_batches)

    # list_batches += list_batches

    #################
    # Training
    ################
    lu.print_start_training()

    if settings.scheduled_training:
        sched_str = "scheduled"
    else:
        sched_str = "non_scheduled"

    if settings.augment:
        augment_str = "augment"
    else:
        augment_str = "non_augment"

    log_dir = "torch_log/%s_%s/" % (augment_str, sched_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if settings.augment:
        d_aug = {
            "identity":{},
            "histogram_equalization":{},
            "random_blur":{"max_kernel_size":5},
            "random_erode":{"max_kernel_size":3},
            "random_dilate":{"max_kernel_size":3},
            # "hflip":{},
            # "random_occlusion":{"size_x":30, "size_y":30},
            # "pixel_dropout":{"dropout_fraction":0.05},
            # "equalize_invert":{},
        }

    else:
        d_aug = {
            "identity":{},
        }

    aug = batch_utils.AugDataGenerator(settings, d_aug)
    queue = aug.create_queue()

    try:

        for e in range(settings.nb_epoch):

            s = time()

            d_loss = {0: [], 1:[], 2:[]}

            if settings.scheduled_training:

                if e == 0:
                    model_parameters = [{'params': list_models[0].parameters()}]
                    optimizer = optim.Adam(model_parameters, lr=settings.learning_rate)
                    list_models_epoch = list_models[:1]

                if e == 20:
                    model_parameters = [{'params': list_models[i].parameters()} for i in range(2)]
                    optimizer = optim.Adam(model_parameters, lr=settings.learning_rate)
                    list_models_epoch = list_models[:2]

                if e == 40:
                    model_parameters = [{'params': list_models[i].parameters()} for i in range(3)]
                    optimizer = optim.Adam(model_parameters, lr=settings.learning_rate)
                    list_models_epoch = list_models[:3]
            else:
                model_parameters = [{'params': list_models[i].parameters()} for i in range(3)]
                optimizer = optim.Adam(model_parameters, lr=settings.learning_rate)
                list_models_epoch = list_models

            for i in tqdm(range(settings.n_batch_per_epoch)):

                img0_batch, pts0_batch, init0_batch, delta0_norm_batch = queue.get()

                delta0_norm_var, list_delta0_norm_pred, list_pts0_pred = tu.forward_pass(settings,
                                                                                         list_models_epoch,
                                                                                         img0_batch,
                                                                                         pts0_batch,
                                                                                         init0_batch,
                                                                                         delta0_norm_batch)
                if settings.scheduled_training:
                    if e < 20:
                        l0 = loss_fn(list_delta0_norm_pred[0], delta0_norm_var)
                        d_loss[0].append(l0.cpu().data.numpy())
                        total_loss = l0

                    elif 20 <= e < 40:
                        l0 = loss_fn(list_delta0_norm_pred[0], delta0_norm_var)
                        l1 = loss_fn(list_delta0_norm_pred[0] + list_delta0_norm_pred[1], delta0_norm_var)
                        d_loss[0].append(l0.cpu().data.numpy())
                        d_loss[1].append(l1.cpu().data.numpy())
                        total_loss = l1

                    elif 40 <= e:
                        l0 = loss_fn(list_delta0_norm_pred[0], delta0_norm_var)
                        l1 = loss_fn(list_delta0_norm_pred[0] + list_delta0_norm_pred[1], delta0_norm_var)
                        l2 = loss_fn(list_delta0_norm_pred[0] + list_delta0_norm_pred[1] +
                                     list_delta0_norm_pred[2], delta0_norm_var)
                        d_loss[0].append(l0.cpu().data.numpy())
                        d_loss[1].append(l1.cpu().data.numpy())
                        d_loss[2].append(l2.cpu().data.numpy())
                        total_loss = l2

                else:
                    l0 = loss_fn(list_delta0_norm_pred[0], delta0_norm_var)
                    l1 = loss_fn(list_delta0_norm_pred[0] + list_delta0_norm_pred[1], delta0_norm_var)
                    l2 = loss_fn(list_delta0_norm_pred[0] + list_delta0_norm_pred[1] +
                                 list_delta0_norm_pred[2], delta0_norm_var)
                    d_loss[0].append(l0.cpu().data.numpy())
                    d_loss[1].append(l1.cpu().data.numpy())
                    d_loss[2].append(l2.cpu().data.numpy())
                    total_loss = l0 + l1 + l2

                if False:

                    import matplotlib.pylab as plt
                    import matplotlib.gridspec as gridspec
                    from matplotlib.pyplot import cm
                    gs = gridspec.GridSpec(5, 4)
                    fig = plt.figure(figsize=(15, 15))
                    for i in range(20):
                        ax = plt.subplot(gs[i])
                        ax.imshow(img0_batch[i], cmap="gray")
                    gs.tight_layout(fig)
                    plt.show()
                    plt.clf()
                    plt.close()

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            # Compute eval metrics
            print("")
            for stage in range(len(list_models_epoch)):
                print("Stage : %s, loss: %s" % (stage, np.mean(d_loss[stage])))

            d_losses["train_losses"].append(np.mean(d_loss[len(list_models_epoch) - 1]))
            d_losses["duration"].append(time() - s)
            d_losses, ri = tu.display_metrics(settings, d_losses, list_models_epoch,
                                              tr_img0, tr_pts0, tr_init0,
                                              val_img0, val_pts0, val_init0)

            with open(os.path.join(log_dir, "d_losses.json"), "w") as f:
                for key in d_losses.keys():
                    d_losses[key] = map(float, d_losses[key])
                json.dump(d_losses, f)

        # Stop the queues
        aug.stop_queue()

    except KeyboardInterrupt:
        aug.stop_queue()
