import h5py
import torch
import numpy as np
from torch.nn.functional import affine_grid, grid_sample
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import cm




import cv2
img = cv2.equalizeHist(img)
img = img.astype(np.float32)

plt.figure()
plt.imshow(img[:,:], cmap="gray")


img = np.reshape(img, (1,1,240,240))

theta1 = np.zeros((2,3))
theta1[0, 0] = 120 / 240.
theta1[1, 1] = 120 / 240.
theta1[0,2] = -0.5
theta1[1,2] = -0.5

theta2 = np.zeros((2,3))
theta2[0, 0] = 0.5
theta2[1, 1] = 0.5


theta = np.expand_dims(theta1, 0)
size = torch.Size((1, 1, 120,120))

theta = torch.FloatTensor(theta)
img = torch.FloatTensor(img)

grid = affine_grid(theta, size)

out_img = grid_sample(img, grid)

plt.figure()
plt.imshow(out_img[0,0,:,:].data.numpy(), cmap="gray")
plt.show()


# plt.imshow(np.concatenate((img, img2), axis=1), cmap="gray")
# plt.scatter(pts[:, 0], pts[:, 1], s=5)
# plt.scatter(pts2[:, 0] + 240, pts2[:, 1], s=5)
# plt.show()
# plt.clf()
# plt.close()

################
# numpy version
################


patch_shape = (70, 70)
patch_shape = np.array(patch_shape)
patch_half_shape = np.require(np.round(patch_shape / 2), dtype=int)
start = -patch_half_shape
end = patch_half_shape
sampling_grid = np.mgrid[start[0]:end[0], start[1]:end[1]]
sampling_grid = sampling_grid.swapaxes(0, 2).swapaxes(0, 1)


max_x = img.shape[0] - 1
max_y = img.shape[1] - 1

patch_grid = (sampling_grid[None, :, :, :] + pts[:, None, None, :]).astype('int32')

X = patch_grid[:, :, :, 0].clip(0, max_x)
Y = patch_grid[:, :, :, 1].clip(0, max_y)

patches = img[Y, X].transpose(0,2,1)

patch_grid2 = (sampling_grid[None, :, :, :] + pts2[:, None, None, :]).astype('int32')

X = patch_grid2[:, :, :, 0].clip(0, max_x)
Y = patch_grid2[:, :, :, 1].clip(0, max_y)

patches2 = img2[Y, X].transpose(0,2,1)


# patches_T = img[Y,X].transpose(0,2,1)

# Plot for debugging

# gs = gridspec.GridSpec(5, 4)
# fig = plt.figure(figsize=(15, 15))
# for i in range(20):
#     ax = plt.subplot(gs[i])
#     ax.imshow(patches[i], cmap="gray")
# gs.tight_layout(fig)
# plt.show()


################
# torch version no batch image
################

img = np.expand_dims(img, 0)
img2 = np.expand_dims(img2, 0)

img = np.concatenate((img, img2), axis=0)

pts = np.expand_dims(pts, 0)
pts2 = np.expand_dims(pts2, 0)
pts = np.concatenate((pts, pts2), axis=0)


img = torch.LongTensor(img.astype(np.int64))
sampling_grid = torch.FloatTensor(sampling_grid.astype(np.float32))
pts = torch.FloatTensor(pts.astype(np.float32))
sampling_grid_size = sampling_grid.size()
pts_size = pts.size()

sg = sampling_grid.contiguous().view((1,1) + sampling_grid_size).expand((pts_size[0], pts_size[1]) + sampling_grid_size)
ppts = pts.contiguous().view((pts_size[0], pts_size[1], 1, 1, pts_size[-1])).expand((pts_size[0], pts_size[1],
                                                                                     sampling_grid_size[1], sampling_grid_size[1], pts_size[-1]))

patch_grid = (sg + ppts).long()


# assert np.all(np.isclose(patch_grid, patch_grid_torch.numpy()))

X = patch_grid[:, :, :, :, 0].clamp(0, max_x)
Y = patch_grid[:, :, :, :, 1].clamp(0, max_y)

# X = X[:, :, 0].contiguous().view(-1)
Y = Y[:, :, 0, :].contiguous().view(2, -1)
Y = Y.contiguous().view(2, -1, 1)
Y = Y.expand(2,21 * 70,240)
img = img.gather(1, Y).contiguous().view(2, 21, 70, 240)

# gs = gridspec.GridSpec(5, 4)
# fig = plt.figure(figsize=(15, 15))
# for i in range(20):
#     ax = plt.subplot(gs[i])
#     img0 = img[0, i, :, :]
#     img1 = img[1, i, :, :]
#     img_plot = np.concatenate((img0, img1), axis=1)
#     ax.imshow(img_plot, cmap="gray")
# gs.tight_layout(fig)
# plt.show()
img_patch = img.gather(3, X.permute(0,1,3,2)).numpy()

patches_numpy = np.concatenate((np.expand_dims(patches, 0), np.expand_dims(patches2, 0)), 0)

gs = gridspec.GridSpec(5, 4)
fig = plt.figure(figsize=(15, 15))
for i in range(20):
    ax = plt.subplot(gs[i])
    img0 = img_patch[0, i, :, :]
    img1 = img_patch[1, i, :, :]

    img0_numpy = patches_numpy[0, i, :, :]
    img1_numpy = patches_numpy[1, i, :, :]

    img_plot = np.concatenate((img0, img1), axis=1)
    img_plot_numpy = np.concatenate((img0_numpy, img1_numpy), axis=1)

    imgout = np.concatenate((img_plot, img_plot_numpy), 0)
    # ax.imshow(imgout, cmap="gray")
    ax.imshow(img_plot - img_plot_numpy, cmap="gray")
gs.tight_layout(fig)
plt.show()


# ################
# # torch version no batch image
# ################

# img = torch.LongTensor(img.astype(np.int64))
# sampling_grid = torch.FloatTensor(sampling_grid.astype(np.float32))
# pts = torch.FloatTensor(pts.astype(np.float32))
# sampling_grid_size = sampling_grid.size()
# pts_size = pts.size()

# import ipdb
# ipdb.set_trace()

# sg = sampling_grid.contiguous().view((1,) + sampling_grid_size).expand((pts_size[0],) + sampling_grid_size)
# ppts = pts.contiguous().view((pts_size[0], 1, 1, pts_size[-1])).expand((pts_size[0],
# sampling_grid_size[1], sampling_grid_size[1], pts_size[-1]))

# patch_grid = (sg + ppts).long()

# # assert np.all(np.isclose(patch_grid, patch_grid_torch.numpy()))

# X = patch_grid[:, :, :, 0].clamp(0, max_x)
# Y = patch_grid[:, :, :, 1].clamp(0, max_y)

# # X = X[:, :, 0].contiguous().view(-1)
# Y = Y[:, 0, :].contiguous().view(-1)
# # import ipdb; ipdb.set_trace()
# # import ipdb; ipdb.set_trace()
# img_patch = img.index_select(0, Y).view(21, 50, 240).gather(2, X.permute(0,2,1)).numpy()


# gs = gridspec.GridSpec(5, 4)
# fig = plt.figure(figsize=(15, 15))
# for i in range(20):
#     ax = plt.subplot(gs[i])
#     img = patches[i] - img_patch[i]
#     img = np.concatenate((patches[i], img_patch[i]), axis=1)
#     ax.imshow(img, cmap="gray")
# gs.tight_layout(fig)
# plt.show()


# import ipdb
# ipdb.set_trace()


# X = X[:, :, :1].repeat(1,1,240)
# Y = Y[:, :, :1].repeat(1,1,50)
# img = img.view(1,240,240).repeat(21,1,1)
# new_img = img.gather(1,X).gather(2,Y).numpy()

# gs = gridspec.GridSpec(5, 4)
# fig = plt.figure(figsize=(15, 15))
# for i in range(20):
#     ax = plt.subplot(gs[i])
#     ax.imshow(new_img[i], cmap="gray")
# gs.tight_layout(fig)
# plt.show()

# import ipdb
# ipdb.set_trace()

# import ipdb
# ipdb.set_trace()
# X = X[:, :, 0].contiguous().view(-1)
# Y = Y[:, :, 1].contiguous().view(-1)

# img_patch = img.index_select(0, Y).index_select(1, X).numpy()

# # xx = X.view(21, -1)
# # yy = Y.view(21, -1)

# Z = torch.cat((xx, yy), dim=1)
# Z = Z.view(-1, 2)

# import ipdb
# ipdb.set_trace()

# img_patch = img.index_select(0, Z[:, 0]).index_select(1, Z[:, 1]).numpy()
# img_patch = img_patch.reshape((21, 50, 50))

# # plt.imshow(img_patch, cmap="gray")
# # plt.show()

# gs = gridspec.GridSpec(5, 4)
# fig = plt.figure(figsize=(15, 15))
# for i in range(20):
#     ax = plt.subplot(gs[i])
#     ax.imshow(img_patch[i], cmap="gray")
# gs.tight_layout(fig)
# plt.show()

# import ipdb
# ipdb.set_trace()

# # import ipdb; ipdb.set_trace()
# # patch_grid = sampling_grid.expand((pts_size[0],) + sampling_grid_size) + pts.expand((pts_size[0], sampling_grid_size[1], sampling_grid_size[2], pts_size[-1]))
# print(sampling_grid.size(), patch_grid.size())
