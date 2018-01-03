import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import cm

# class SubCNN(nn.Module):
#     def __init__(self, input_dim, output_dim, n_filters, hidden_dim):

#         super(SubCNN, self).__init__()

#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.n_filters = n_filters
#         self.hidden_dim = hidden_dim

#         self.mp1 = nn.MaxPool2d(16, 16)

#         self.fc1 = nn.Linear(21 * 4 * 4, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x, batch_size):

#         # Convolutional part
#         x = self.mp1(x)

#         # Flatten
#         x = x.view(batch_size, -1)

#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)

#         return x


class SubCNNsmall(nn.Module):
    def __init__(self, input_dim, output_dim, n_filters, hidden_dim):

        super(SubCNN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_filters = n_filters
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Conv2d(input_dim, 16, 7, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 2)

        self.fc1 = nn.Linear(32 * 2 * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, batch_size):

        # Convolutional part
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten
        x = x.view(batch_size, -1)

        # Fully connected part
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# network
class SubCNN(nn.Module):
    def __init__(self, input_dim, output_dim, n_filters, hidden_dim):

        super(SubCNN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_filters = n_filters
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Conv2d(input_dim, 32, 3, 1, padding=(1,1))
        self.conv2 = nn.Conv2d(32, 32, 3, 1, padding=(1,1))
        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=(1,1))
        self.conv4 = nn.Conv2d(64, 64, 3, 1, padding=(1,1))
        self.conv5 = nn.Conv2d(64, 128, 3, 1, padding=(1,1))
        self.conv6 = nn.Conv2d(128, 128, 3, 1, padding=(1,1))

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.mp1 = nn.MaxPool2d(2, 2)
        self.mp2 = nn.MaxPool2d(2, 2)
        self.mp3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 4 * 4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, batch_size):

        # Convolutional part
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.mp1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.mp2(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.mp3(x)

        # Flatten
        x = x.view(batch_size, -1)

        # Fully connected part
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class SubCNNnew(nn.Module):
    def __init__(self, input_dim, output_dim, n_filters, hidden_dim):

        super(SubCNNnew, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_filters = n_filters
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Conv2d(1, 32, 6, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 3, 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(2 * 672, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, batch_size):

        x = x.view(x.size(0), x.size(1), 1, x.size(2), x.size(3))

        list_out = []
        for i in range(x.size(1)):
            # Convolutional part
            xtemp = F.relu(self.bn1(self.conv1(x[:, i, :, :, :])))
            xtemp = self.maxpool1(xtemp)
            xtemp = F.relu(self.bn2(self.conv2(xtemp)))
            xtemp = self.maxpool2(xtemp)
            list_out.append(xtemp)

        x = torch.cat(list_out, 1)

        # Flatten
        x = x.view(batch_size, -1)

        # Fully connected part
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class SubCNNSTN(nn.Module):
    def __init__(self, input_dim, output_dim, n_filters, hidden_dim):

        super(SubCNNSTN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_filters = n_filters
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Conv2d(input_dim, 32, 3, 1, padding=(1,1))
        self.conv2 = nn.Conv2d(32, 32, 3, 1, padding=(1,1))
        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=(1,1))
        self.conv4 = nn.Conv2d(64, 64, 3, 1, padding=(1,1))
        self.conv5 = nn.Conv2d(input_dim, 32, 3, 1, padding=(1,1))
        self.conv6 = nn.Conv2d(32, 32, 3, 1, padding=(1,1))

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)
        self.bn6 = nn.BatchNorm2d(32)

        self.mp1 = nn.MaxPool2d(2, 2)
        self.mp2 = nn.MaxPool2d(2, 2)
        self.mp3 = nn.MaxPool2d(2, 2)

        self.ST_fc1 = nn.Linear(64 * 8 * 8, hidden_dim)
        self.ST_fc2 = nn.Linear(hidden_dim, 6)

        self.fc1 = nn.Linear(8192, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, batch_size):

        # STN part
        x_ST = F.relu(self.bn1(self.conv1(x)))
        x_ST = F.relu(self.bn2(self.conv2(x_ST)))
        x_ST = self.mp1(x_ST)
        x_ST = F.relu(self.bn3(self.conv3(x_ST)))
        x_ST = F.relu(self.bn4(self.conv4(x_ST)))
        x_ST = self.mp2(x_ST)

        x_ST = x_ST.view(batch_size, -1)
        x_ST = F.relu(self.ST_fc1(x_ST))

        theta = self.ST_fc2(x_ST).view(-1,2,3)
        out_size = torch.Size((batch_size, x.size(1), x.size(2), x.size(3)))

        grid = torch.nn.functional.affine_grid(theta, out_size).clamp(-1,1)
        patches = torch.nn.functional.grid_sample(x, grid)

        # Conv part
        x_out = F.relu(self.bn5(self.conv5(patches)))
        x_out = F.relu(self.bn6(self.conv6(x_out)))
        x_out = self.mp3(x_out)

        # Flatten
        x_out = x_out.view(batch_size, -1)

        # Fully connected part
        x_out = F.relu(self.fc1(x_out))
        x_out = self.fc2(x_out)

        idx = np.random.uniform(0, 1000)
        if idx > 999.999:

            patches_numpy = patches.cpu().data.numpy()[0]
            img_numpy = x.cpu().data.numpy()[0]

            gs = gridspec.GridSpec(5, 4)
            fig = plt.figure(figsize=(15, 15))
            for i in range(20):
                ax = plt.subplot(gs[i])
                im1 = img_numpy[i]
                im2 = patches_numpy[i]
                im = np.concatenate((im1, im2), axis=1)
                ax.imshow(im, cmap="gray")
            gs.tight_layout(fig)
            plt.savefig("%s.png" % idx)
            plt.clf()
            plt.close()

        return x_out


class localizerCNN(nn.Module):
    def __init__(self):

        super(localizerCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 2)
        self.conv2 = nn.Conv2d(32, 32, 3, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, 2)
        self.conv5 = nn.Conv2d(64, 64, 3, 2)

        self.conv6 = nn.Conv2d(1, 32, 3, 2)
        self.conv7 = nn.Conv2d(32, 32, 3, 2)
        self.conv8 = nn.Conv2d(32, 32, 3, 2)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(32)
        self.bn8 = nn.BatchNorm2d(32)

        self.ST_fc1 = nn.Linear(64 * 6 * 6, 128)
        self.ST_fc2 = nn.Linear(128, 6)

        self.fc1 = nn.Linear(32 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 42)

    def forward(self, x, batch_size):

        # STN part
        x_ST = F.relu(self.bn1(self.conv1(x)))
        x_ST = F.relu(self.bn2(self.conv2(x_ST)))
        x_ST = F.relu(self.bn3(self.conv3(x_ST)))
        x_ST = F.relu(self.bn4(self.conv4(x_ST)))
        x_ST = F.relu(self.bn5(self.conv5(x_ST)))

        x_ST = x_ST.view(batch_size, -1)
        x_ST = F.relu(self.ST_fc1(x_ST))

        # eye = Variable(0.25 * torch.eye(2)).cuda()
        # eye = eye.view(1,2,2).expand(batch_size, 2, 2)

        theta = self.ST_fc2(x_ST).view(-1,2,3).clamp(-1,1)
        # theta = torch.cat((eye, theta), -1).clamp(-1,1)

        out_size = torch.Size((batch_size, x.size(1), x.size(2) // 4, x.size(3) // 4))

        grid = torch.nn.functional.affine_grid(theta, out_size).clamp(-1,1)
        ST_out = torch.nn.functional.grid_sample(x, grid)

        # Conv part
        x_out = F.relu(self.bn6(self.conv6(ST_out)))
        x_out = F.relu(self.bn7(self.conv7(x_out)))
        x_out = F.relu(self.bn8(self.conv8(x_out)))

        # Flatten
        x_out = x_out.view(batch_size, -1)

        # Fully connected part
        x_out = F.relu(self.fc1(x_out))
        x_out = self.fc2(x_out)

        return x_out, ST_out
