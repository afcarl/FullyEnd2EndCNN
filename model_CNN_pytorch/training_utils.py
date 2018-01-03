import h5py
import numpy as np
from time import time
from tqdm import tqdm
import cv2
import torch
from torch.autograd import Variable
try:
    from Queue import Queue
    from Queue import Empty
except ImportError:
    from queue import Queue
    from queue import Empty
from ..utils import logging_utils as lu

try:
    from torch.nn.functional import grid_sample
except ImportError:
    pass

import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import cm
import os


def mse_loss(input, target):
    return torch.sum(torch.pow(input - target, 2)) / input.data.nelement()


def cyclic_lr(optimizer, it, step_size, base_lr, max_lr, gamma):

    cycle = np.floor(1 + it / (2. * step_size))
    x = np.abs(it / float(step_size) - 2. * cycle + 1)
    desired_lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) * gamma**(it)

    for param_group in optimizer.param_groups:
        param_group['lr'] = desired_lr

    return optimizer


def unnormalize(X, X_mean, X_std):

    return X * X_std + X_mean


def get_data_shapes_and_norm(hdf5_file, iteration):

    with h5py.File(hdf5_file, "r") as hf:
        X_shape = hf["training/it_%s/samples_feat" % iteration].shape
        y_shape = hf["training/orig_params"].shape
        mean_delta_params = hf["norm/it_%s/mean_delta_params" % iteration][:]
        std_delta_params = hf["norm/it_%s/std_delta_params" % iteration][:]

        lu.print_shape(X_shape, y_shape)

        return X_shape, y_shape, mean_delta_params, std_delta_params


def load_normalizations(settings, hdf5_file):

    with h5py.File(hdf5_file, "r") as hf:

        # Normalize delta
        delta0_mean = hf["delta0_mean"][:]
        delta0_std = hf["delta0_std"][:]

    settings.delta0_mean = delta0_mean
    settings.delta0_std = delta0_std

    settings.delta0_mean_tensor = torch.FloatTensor(delta0_mean).cuda()
    settings.delta0_std_tensor = torch.FloatTensor(delta0_std).cuda()


def _mmap_h5(path, h5path):

    with h5py.File(path) as f:
        ds = f[h5path]
        # We get the dataset address in the HDF5 field.
        offset = ds.id.get_offset()
        # We ensure we have a non-compressed contiguous array.
        assert ds.chunks is None
        assert ds.compression is None
        assert offset > 0
        dtype = ds.dtype
        shape = ds.shape
    arr = np.memmap(path, mode="r", shape=shape, offset=offset, dtype=dtype)
    return arr


def load_data(hdf5_file, dset_type, size=None):

    img0 = _mmap_h5("data/processed/CNN_database.h5", "%s/img0" % dset_type)
    pts0 = _mmap_h5("data/processed/CNN_database.h5", "%s/pts0" % dset_type)
    init0 = _mmap_h5("data/processed/CNN_database.h5", "%s/init0" % dset_type)

    if size is None:

        img0 = np.array(img0[:]).astype(np.float32) / 255.
        pts0 = np.array(pts0[:]).astype(np.float32)
        init0 = np.array(init0[:]).astype(np.float32)

    else:

        idxs = np.random.choice(img0.shape[0], size)

        img0 = np.array(img0[idxs]).astype(np.float32) / 255.
        pts0 = np.array(pts0[idxs]).astype(np.float32)
        init0 = np.array(init0[idxs]).astype(np.float32)

    return img0, pts0, init0


def forward_pass(settings, list_models, img0, pts0, init0, delta0_norm):

    ####################################
    # wrap to float tensor and variable
    ######################################
    max_x = settings.max_x
    max_y = settings.max_y

    patch_size = settings.patch_size
    batch_size = img0.shape[0]

    # Images
    img0_var = Variable(torch.FloatTensor(img0).cuda())
    # Starting landmarks (-1, 21, 2), normalized between 0 and 1
    init0_var = Variable(torch.FloatTensor(init0).cuda())
    # target delta (-1, 42), normalized (mean and std)
    delta0_norm_var = Variable(torch.FloatTensor(delta0_norm).cuda())

    # Sampling grid (reshape it to have the same size as batch size)
    batch_size = init0_var.size(0)

    # Normalization params
    delta0_mean = Variable(settings.delta0_mean_tensor.expand(batch_size, 42))
    delta0_std = Variable(settings.delta0_std_tensor.expand(batch_size, 42))

    pts0_pred_var = None

    eye = float(patch_size) / 240. * torch.eye(2).cuda()
    eye = eye.view(1,1,2,2).expand(batch_size, 21, 2, 2)

    ##################################
    # Computational graph below
    ###################################

    list_delta0_norm_pred = []
    list_pts0_pred = []

    # Loop over models  SAMPLING GRID
    for model_idx in range(len(list_models)):

        if model_idx == 0:
            theta_params = init0_var.view(init0_var.size() + (1,)) / 120. - 1
            theta = torch.cat((eye, theta_params.data), -1)
        else:
            theta_params = pts0_pred_var.view(init0_var.size() + (1,)).view(init0_var.size() + (1,)) / 120. - 1
            theta = torch.cat((eye, theta_params.data), dim=-1)

        list_grid = []
        for i in range(theta.size(1)):
            theta_patch = theta[:, i, :, :].contiguous()
            out_size = torch.Size((batch_size, 1, patch_size, patch_size))
            grid = torch.nn.functional.affine_grid(theta_patch, out_size).clamp(-1,1)
            list_grid.append(grid)

        s = img0_var.size()
        im0_var = img0_var.view(s[0], 1, s[2], s[2])
        list_out = []
        for idx in range(len(list_grid)):
            list_out.append(grid_sample(im0_var, list_grid[idx]))
        patches = torch.cat(list_out, 1)

        if False:
            patches = patches.cpu().data.numpy()

            gs = gridspec.GridSpec(5, 4)
            fig = plt.figure(figsize=(15, 15))
            for i in range(20):
                ax = plt.subplot(gs[i])
                ax.imshow(patches[0, i], cmap="gray")
            gs.tight_layout(fig)
            plt.show()

        delta0_norm_pred = list_models[model_idx](patches, batch_size)
        # Unnormalize delta0_pred
        delta0_pred = delta0_norm_pred * delta0_std + delta0_mean
        # Get new predicted landmarks
        if model_idx == 0:
            pts0_pred_var = delta0_pred + init0_var.view(delta0_pred.size())
        else:
            pts0_pred_var = delta0_pred + pts0_pred_var

        list_delta0_norm_pred.append(delta0_norm_pred)
        list_pts0_pred.append(pts0_pred_var)

    return delta0_norm_var, list_delta0_norm_pred, list_pts0_pred


def mean_and_std_shape_rmse(gt, pred):

    rmse = np.power(gt - pred, 2)
    rmse = np.mean(rmse, axis=1)
    rmse = np.sqrt(rmse)

    rmse_mean = np.mean(rmse, axis=0)
    rmse_std = np.std(rmse, axis=0)

    return rmse_mean, rmse_std


def get_metrics_inmemory(settings, list_models, img0, pts0, init0):

    num_elem = img0.shape[0]
    chunk_size = settings.batch_size
    num_chunks = num_elem / chunk_size
    list_chunks = np.array_split(np.arange(num_elem), num_chunks)
    list_pred = []

    for batch_idxs in list_chunks:

        start = batch_idxs[0]
        end = batch_idxs[-1]

        img0_batch = img0[start: end + 1]
        pts0_batch = pts0[start: end + 1]
        # Select a random init
        idx_init = np.random.randint(0, 100)
        init0_batch = init0[start: end + 1, idx_init, :, :]

        delta0_norm_batch = (pts0_batch - init0_batch).reshape(-1, 42)
        delta0_norm_batch -= settings.delta0_mean
        delta0_norm_batch /= settings.delta0_std

        _, _, list_pts0_pred = forward_pass(settings, list_models,
                                            img0_batch,
                                            pts0_batch,
                                            init0_batch,
                                            delta0_norm_batch)

        list_pred.append(list_pts0_pred[-1].cpu().data.numpy())

    pts0_pred = np.concatenate(list_pred, axis=0)
    pts0 = pts0.reshape(pts0_pred.shape)

    rmse_mean, rmse_std = mean_and_std_shape_rmse(pts0, pts0_pred)

    return rmse_mean, rmse_std


def display_metrics(settings, d_losses, list_models,
                    tr_img0, tr_pts0, tr_init0,
                    val_img0, val_pts0, val_init0):

    s = time()

    # Compute training loss
    idxs_train = np.random.permutation(tr_img0.shape[0])[:val_img0.shape[0]]  # subset of training for speed
    out = get_metrics_inmemory(settings, list_models,
                               tr_img0[idxs_train],
                               tr_pts0[idxs_train],
                               tr_init0[idxs_train])
    rmse_mean_train, rmse_std_train = out
    d_losses["train_mean_shape_rmse"].append(rmse_mean_train)
    d_losses["train_std_shape_rmse"].append(rmse_std_train)

    # Compute validation loss
    out = get_metrics_inmemory(settings, list_models,
                               val_img0,
                               val_pts0,
                               val_init0)
    rmse_mean_val, rmse_std_val = out

    d_losses["val_mean_shape_rmse"].append(rmse_mean_val)
    d_losses["val_std_shape_rmse"].append(rmse_std_val)

    # Special case, first training iteration
    if len(d_losses["val_mean_shape_rmse"]) == 1:
        d_losses["best_val_mean_shape_rmse"].append(d_losses["val_mean_shape_rmse"][-1])
        # Arbitrary value for relative improvement
        ri = 100

    else:
        if d_losses["val_mean_shape_rmse"][-1] < d_losses["best_val_mean_shape_rmse"][-1]:
            d_losses["best_val_mean_shape_rmse"].append(d_losses["val_mean_shape_rmse"][-1])
        else:
            d_losses["best_val_mean_shape_rmse"].append(d_losses["best_val_mean_shape_rmse"][-1])
        # Compute relative improvement
        ri = (d_losses["best_val_mean_shape_rmse"][-2] / d_losses["best_val_mean_shape_rmse"][-1] - 1) * 100

    list_tuples = []
    epoch_idx = len(d_losses["val_mean_shape_rmse"]) - 1
    d_losses["duration"][epoch_idx] += time() - s
    list_tuples.append(tuple([epoch_idx,
                              "%.4g" % d_losses["train_losses"][epoch_idx],
                              "%.4g" % d_losses["train_mean_shape_rmse"][epoch_idx],
                              "%.4g" % d_losses["train_std_shape_rmse"][epoch_idx],
                              "%.4g" % d_losses["val_mean_shape_rmse"][epoch_idx],
                              "%.4g" % d_losses["val_std_shape_rmse"][epoch_idx],
                              "%.4g" % d_losses["best_val_mean_shape_rmse"][epoch_idx],
                              str(int(d_losses["duration"][epoch_idx]))])
                       )

    TABLE_DATA = (("Epoch", "T MSE",
                   "T M_S_RMSE", "T STD_S_RMSE",
                   "V M_S_RMSE", "V STD_S_RMSE",
                   "V Best M_S_RMSE",
                   "Time (s)"),)

    TABLE_DATA += tuple(list_tuples)

    lu.print_table(TABLE_DATA)

    return d_losses, ri


def plot_regression(settings, list_models, epoch,
                    img0, pts0, init0,
                    dset_type):

    # Select first init and get the normalized delta0
    idx_init = 0
    init0 = init0[:, idx_init, :, :]
    delta0_norm = (pts0 - init0).reshape(-1, 42)
    delta0_norm -= settings.delta0_mean
    delta0_norm /= settings.delta0_std

    _, _, pts0_preds = forward_pass(settings,
                                    list_models,
                                    img0,
                                    pts0,
                                    init0,
                                    delta0_norm)

    list_pts0_pred = [init0]
    for p in pts0_preds:
        p = p.cpu().data.numpy().reshape(-1, 21, 2)
        list_pts0_pred.append(p)

    for k in range(max(4, img0.shape[0])):

        fig_dir = "figures_pytorch/%s/fig_%s" % (dset_type, k)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        img0_plot = img0[k]
        pts0_plot = pts0[k]

        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2,2,hspace=0.01)
        for i in range(len(list_pts0_pred)):

            pts0_pred_plot = list_pts0_pred[i][k]

            ax = plt.subplot(gs[i])
            ax.imshow(img0_plot, cmap="gray")
            ax.scatter(pts0_plot[:, 0], pts0_plot[:, 1], s=10, color="C0")
            ax.scatter(pts0_pred_plot[:, 0], pts0_pred_plot[:, 1], s=10, color="C1")

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("Iteration %s" % i, fontsize=18)
        gs.tight_layout(fig)
        plt.savefig(os.path.join(fig_dir, "epoch_%s.png" % epoch))
        plt.clf()
        plt.close()


def plot_localizer(settings, model, epoch, img0, pts0, dset_type):

    batch_size = img0.shape[0]

    # Make predictions
    img0_var = Variable(torch.FloatTensor(np.expand_dims(img0, 1)).cuda())
    pts0_pred, img0_loc = model(img0_var, batch_size)

    pts0_pred = pts0_pred.cpu().data.numpy()
    img0_loc = img0_loc.cpu().data.numpy()[:, 0, :, :]

    # Unnormalize pts0_pred
    pts0_pred = (pts0_pred + 1.) * 120.
    pts0_pred = pts0_pred.reshape(pts0.shape)

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(4,4)

    for k in range(min(16, batch_size)):

        fig_dir = "figures_pytorch_localizer/%s" % dset_type
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        img0_plot = img0[k]
        pts0_plot = pts0[k]
        pts0_pred_plot = pts0_pred[k]
        img0_loc_plot = img0_loc[k]

        h, w = img0_plot.shape

        img0_loc_plot = cv2.resize(img0_loc_plot, (w, h))

        img_plot = np.concatenate((img0_plot, img0_loc_plot), axis=1)

        ax = plt.subplot(gs[k])
        ax.imshow(img_plot, cmap="gray")
        ax.scatter(pts0_plot[:, 0], pts0_plot[:, 1], s=10, color="C0")
        ax.scatter(pts0_pred_plot[:, 0], pts0_pred_plot[:, 1], s=10, color="C1")

        ax.set_xticks([])
        ax.set_yticks([])
    gs.tight_layout(fig)
    plt.savefig(os.path.join(fig_dir, "epoch_%s.png" % epoch))
    plt.clf()
    plt.close()


def get_metrics_inmemory_localizer(settings, model, img0, pts0):

    num_elem = img0.shape[0]
    chunk_size = settings.batch_size
    num_chunks = num_elem / chunk_size
    list_chunks = np.array_split(np.arange(num_elem), num_chunks)
    list_pred = []

    for batch_idxs in list_chunks:

        start = batch_idxs[0]
        end = batch_idxs[-1]

        img0_batch = img0[start: end + 1]
        # Images
        img0_var = Variable(torch.FloatTensor(np.expand_dims(img0_batch, 1)).cuda())
        pts0_pred, _ = model(img0_var, img0_batch.shape[0])

        pts0_pred = pts0_pred.cpu().data.numpy()
        pts0_pred = (pts0_pred + 1.) * 120

        list_pred.append(pts0_pred)

    pts0_pred = np.concatenate(list_pred, axis=0)
    pts0 = pts0.reshape(pts0_pred.shape)

    rmse_mean, rmse_std = mean_and_std_shape_rmse(pts0, pts0_pred)

    return rmse_mean, rmse_std


def display_metrics_localizer(settings, d_losses, model,
                              tr_img0, tr_pts0,
                              val_img0, val_pts0):

    s = time()

    # Compute training loss
    idxs_train = np.random.permutation(tr_img0.shape[0])[:val_img0.shape[0]]  # subset of training for speed
    out = get_metrics_inmemory_localizer(settings, model,
                                         tr_img0[idxs_train],
                                         tr_pts0[idxs_train])
    rmse_mean_train, rmse_std_train = out
    d_losses["train_mean_shape_rmse"].append(rmse_mean_train)
    d_losses["train_std_shape_rmse"].append(rmse_std_train)

    # Compute validation loss
    out = get_metrics_inmemory_localizer(settings, model,
                                         val_img0,
                                         val_pts0)
    rmse_mean_val, rmse_std_val = out

    d_losses["val_mean_shape_rmse"].append(rmse_mean_val)
    d_losses["val_std_shape_rmse"].append(rmse_std_val)

    # Special case, first training iteration
    if len(d_losses["val_mean_shape_rmse"]) == 1:
        d_losses["best_val_mean_shape_rmse"].append(d_losses["val_mean_shape_rmse"][-1])
        # Arbitrary value for relative improvement
        ri = 100

    else:
        if d_losses["val_mean_shape_rmse"][-1] < d_losses["best_val_mean_shape_rmse"][-1]:
            d_losses["best_val_mean_shape_rmse"].append(d_losses["val_mean_shape_rmse"][-1])
        else:
            d_losses["best_val_mean_shape_rmse"].append(d_losses["best_val_mean_shape_rmse"][-1])
        # Compute relative improvement
        ri = (d_losses["best_val_mean_shape_rmse"][-2] / d_losses["best_val_mean_shape_rmse"][-1] - 1) * 100

    list_tuples = []
    epoch_idx = len(d_losses["val_mean_shape_rmse"]) - 1
    d_losses["duration"][epoch_idx] += time() - s
    list_tuples.append(tuple([epoch_idx,
                              "%.4g" % d_losses["train_losses"][epoch_idx],
                              "%.4g" % d_losses["train_mean_shape_rmse"][epoch_idx],
                              "%.4g" % d_losses["train_std_shape_rmse"][epoch_idx],
                              "%.4g" % d_losses["val_mean_shape_rmse"][epoch_idx],
                              "%.4g" % d_losses["val_std_shape_rmse"][epoch_idx],
                              "%.4g" % d_losses["best_val_mean_shape_rmse"][epoch_idx],
                              str(int(d_losses["duration"][epoch_idx]))])
                       )

    TABLE_DATA = (("Epoch", "T MSE",
                   "T M_S_RMSE", "T STD_S_RMSE",
                   "V M_S_RMSE", "V STD_S_RMSE",
                   "V Best M_S_RMSE",
                   "Time (s)"),)

    TABLE_DATA += tuple(list_tuples)

    lu.print_table(TABLE_DATA)

    return d_losses, ri
