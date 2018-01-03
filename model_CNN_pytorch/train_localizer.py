import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable

from tqdm import tqdm
from .models import localizerCNN
from . import training_utils as tu
from ..utils import logging_utils as lu

from time import time
from time import sleep
from . import batch_utils
import shutil


def train(settings):
    """Train NN for eye tracking regression

    Args:
        settings: StereoTrackerSettings instance
        iteration: Iteration of the regression cascade
    """

    try:
        shutil.rmtree("figures_pytorch_localizer")
    except Exception:
        pass

    ###########################
    # Data and normalization
    ###########################

    # Get data
    val_img0, val_pts0, _ = tu.load_data("data/processed/CNN_database.h5", "validation")
    tr_img0, tr_pts0, _ = tu.load_data("data/processed/CNN_database.h5", "training", size=val_img0.shape[0])

    # Get normalizations (stored in settings)
    tu.load_normalizations(settings,"data/processed/CNN_database.h5")

    # Random idx to plot figures.
    idxs_train = np.random.permutation(tr_img0.shape[0])[:16]
    idxs_val = np.random.permutation(val_img0.shape[0])[:16]

    ###########################
    # Neural Net
    ###########################
    model = localizerCNN().cuda()
    print(model)

    loss_fn = torch.nn.MSELoss(size_average=True).cuda()

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

    #################
    # Training
    ################
    lu.print_start_training()

    d_aug = {
        "identity":{},
        # "histogram_equalization":{},
        # "hflip":{},
        # "random_blur":{"max_kernel_size":5},
        # "random_erode":{"max_kernel_size":3},
        # "random_dilate":{"max_kernel_size":3},
        # "random_occlusion":{"size_x":30, "size_y":30},
        # "pixel_dropout":{"dropout_fraction":0.05},
        # "equalize_invert":{},
    }

    aug = batch_utils.AugDataGenerator(settings, d_aug)
    queue = aug.create_queue()

    # sleep(1)
    # img0_batch, pts0_batch, _, _ = queue.get()
    # aug.stop_queue()

    try:

        for e in range(settings.nb_epoch):

            s = time()

            list_loss = []

            model_parameters = [{'params': model.parameters()}]
            optimizer = optim.Adam(model_parameters, lr=settings.learning_rate)

            for i in tqdm(range(settings.n_batch_per_epoch)):

                img0_batch, pts0_batch, _, _ = queue.get()
                pts0_norm_batch = pts0_batch / 120. - 1

                # Images
                img0_var = Variable(torch.FloatTensor(np.expand_dims(img0_batch, 1)).cuda())
                # target position (-1, 42), normalized [-1, 1]
                pts0_norm_var = Variable(torch.FloatTensor(pts0_norm_batch).cuda())

                pts0_norm_pred, _ = model(img0_var, img0_batch.shape[0])

                total_loss = loss_fn(pts0_norm_pred, pts0_norm_var)
                list_loss.append(total_loss.cpu().data.numpy())

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            # Compute eval metrics
            d_losses["train_losses"].append(np.mean(list_loss))
            d_losses["duration"].append(time() - s)
            d_losses, ri = tu.display_metrics_localizer(settings, d_losses, model,
                                                        tr_img0, tr_pts0,
                                                        val_img0, val_pts0)

            # Plot some images
            if e % 2 == 0:
                # Training set
                tu.plot_localizer(settings, model, e,
                                  tr_img0[idxs_train],
                                  tr_pts0[idxs_train],
                                  "training")
                tu.plot_localizer(settings, model, e,
                                  val_img0[idxs_val],
                                  val_pts0[idxs_val],
                                  "validation")

        aug.stop_queue()

    except KeyboardInterrupt:
        aug.stop_queue()
