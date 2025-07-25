import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity as cos_sim


class Stein_Reuter10K_SVGD(nn.Module):
    def __init__(self, dim, svgd):
        super(Stein_Reuter10K_SVGD, self).__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.Sigmoid()
        )
        self.dim = dim
        self.Clustering = Stein_IMG(svgd)
        self.fc0 = nn.Linear(2 * self.dim, self.dim)


        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, epoch, pre_iter, x, pretrain=False, prune=False, shringage=False):
        z = self.encoder(x)
        quantized, gamma = self.Clustering(epoch, pre_iter, z, pretrain, prune, shringage)
        ## regular training
        double_z = torch.cat((z, quantized), 1)
        recon_z = self.fc0(double_z)
        #recon_z = self.reparametrize(recon_z)
        recon_x = self.decoder(recon_z)
        return z, quantized, gamma, recon_x

class Stein_ImanetNet_SVGD(nn.Module):
    def __init__(self, dim, svgd):
        super(Stein_ImanetNet_SVGD, self).__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(128, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 128),
        )
        self.dim = dim
        self.Clustering = Stein_IMG(svgd)
        self.fc0 = nn.Linear(2 * self.dim, self.dim)


        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, epoch, pre_iter, x, pretrain=False, prune=False, shringage=False):
        z = self.encoder(x)
        quantized, gamma = self.Clustering(epoch, pre_iter, z, pretrain, prune, shringage)

        ## regular training
        double_z = torch.cat((z, quantized), 1)
        recon_z = self.fc0(double_z)
        #recon_z = self.reparametrize(recon_z)
        recon_x = self.decoder(recon_z)
        return z, quantized, gamma, recon_x

class Stein_MNIST_SVGD(nn.Module):
    def __init__(self, dim, svgd):
        super(Stein_MNIST_SVGD, self).__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(784, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 784),
            nn.Sigmoid()
        )
        self.dim = dim
        self.Clustering = Stein_IMG(svgd)
        self.fc0 = nn.Linear(2 * self.dim, self.dim)


        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, epoch, pre_iter, x, pretrain=False, prune=False, shringage=False):
        x = x.view(-1, 784)
        z = self.encoder(x)
        quantized, gamma = self.Clustering(epoch, pre_iter, z, pretrain, prune, shringage)

        ## regular training
        double_z = torch.cat((z, quantized), 1)
        recon_z = self.fc0(double_z)
        #recon_z = self.reparametrize(recon_z)
        recon_x = self.decoder(recon_z)
        return z, quantized, gamma, recon_x

class Stein_IMG(nn.Module):
    def __init__(self, SVGD):
        super(Stein_IMG, self).__init__()
        self.SVGD = SVGD
        self.v = 5
        self.delta = -5
        self.Id_assign = []

    def group_similarity(self):
        Data_Id = np.concatenate(self.Id_assign, axis=0)
        ratio = Data_Id.sum(axis=0)/Data_Id.sum()
        descend_idx = np.argsort(0-ratio) 
        re_Data_id = Data_Id[:, descend_idx]
        group_sim = cos_sim(re_Data_id.T, re_Data_id.T) 
        group_sim = torch.tensor(group_sim, device=self.SVGD.mu.device).float()
        ratio = torch.tensor(ratio, device=self.SVGD.mu.device).float()
        return group_sim, descend_idx, ratio

    def forward(self, epoch, pre_iter, z, flag_pretrain=False, flag_prune=False, flag_shringage=False):
        mu = self.SVGD.mu.detach()
        # Calculate distances
        Dist = torch.unsqueeze(z, 1) - mu
        z_dist = torch.mean(torch.mul(Dist, Dist), 2)
        # for Gaussian
        gamma = F.softmax(self.delta * z_dist, dim=1)
        # for Student t
        # prob = (1 + z_dist / self.v).pow(-(self.v+1)/2)
        # gamma = prob / prob.sum(dim=1).unsqueeze(dim=1)
        #
        ## leverage the gamma in SVGD
        encoding_Idx = torch.argmax(gamma, dim=1).unsqueeze(1)
        Idx = torch.zeros(z.shape[0], mu.shape[0], device=z.device)
        Idx.scatter_(1, encoding_Idx, 1)

        # Quantize and reconstruction
        recons = torch.matmul(Idx, mu)

        if self.training:
            for id in range(5):
                self.SVGD.cluster_step(epoch, z, flag_pretrain)

        if flag_prune:
            self.Id_assign.append(gamma.cpu().data.numpy())
            if flag_shringage:
                group_sim, descend_idx, ratio = self.group_similarity()
                self.SVGD.shrinkage(group_sim, descend_idx, ratio, epoch, pre_iter)
                self.Id_assign = []

        return recons, gamma