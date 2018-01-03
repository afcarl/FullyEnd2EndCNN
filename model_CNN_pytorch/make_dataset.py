import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec


def make_dataset(settings):

    with h5py.File("data/processed/CNN_database.h5", "w") as hfw:

        with h5py.File(settings.raw_data_file, "r") as hf:

            for dset_type in ["training", "validation"][::-1]:

                keys = hf[dset_type].keys()
                np.random.shuffle(keys)

                n_nAg = hf[dset_type][keys[0]]["view0"]["nAg"].shape[0]

                arr_img0 = np.zeros((len(keys), 240, 240), dtype=np.uint8)
                arr_pts0 = np.zeros((len(keys), 21, 2))
                arr_init0 = np.zeros((len(keys), n_nAg, 21, 2))

                for i, key in enumerate(tqdm(keys, "Processing %s data" % dset_type)):

                    pts0 = hf[dset_type][key]["view0"]["ground_pts"][:].reshape((21, 2))
                    nAg0 = hf[dset_type][key]["view0"]["nAg"][:]

                    # extract mean shapes
                    n_mean = np.reshape(settings.PDM.M,(settings.PDM.M.shape[0] / 2,2),order='F')

                    pts0 = np.asarray(pts0)
                    gAn0 = np.linalg.inv(nAg0)

                    g_init0 = np.dot(n_mean, gAn0[:, 0:2, 0:2]).transpose(
                        1,0,2) + np.expand_dims(gAn0[:, 0:2,2], axis=1)

                    arr_init0[i] = g_init0

                    arr_pts0[i] = pts0
                    arr_img0[i] = hf[dset_type][key]["view0"]["image"][:]

                g = hfw.create_group(dset_type)
                g.create_dataset("img0", data=arr_img0)
                g.create_dataset("pts0", data=arr_pts0)
                g.create_dataset("init0", data=arr_init0)

        # Compute norms
        pts0 = np.concatenate((hfw["training/pts0"][:], hfw["validation/pts0"][:]), axis=0)
        init0 = np.concatenate((hfw["training/init0"][:, 0, :, :], hfw["validation/init0"][:, 0, :, :]), axis=0)

        delta0 = (pts0 - init0).reshape(-1, 42)

        hfw.create_dataset("delta0_mean", data=delta0.mean(0).reshape(1, -1).astype(np.float32))
        hfw.create_dataset("delta0_std", data=delta0.std(0).reshape(1, -1).astype(np.float32))


def plot_dataset():

    with h5py.File("data/processed/CNN_database.h5", "r") as hf:

        for dset_type in ["training", "validation"]:

            for i in range(3):

                img0 = hf["%s/img0" % dset_type][i]
                pts0 = hf["%s/pts0" % dset_type][i]
                init0 = hf["%s/init0" % dset_type][i]

                plt.figure(figsize=(20,10))
                gs = gridspec.GridSpec(1,1)
                ax = plt.subplot(gs[0])

                ax.imshow(img0, cmap="gray")
                ax.scatter(pts0[:, 0], pts0[:, 1], label="Truth", s=10)
                ax.scatter(init0[0,:, 0], init0[0,:, 1], label="Start", s=10)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.legend()

                plt.show()
                plt.clf()
                plt.close()