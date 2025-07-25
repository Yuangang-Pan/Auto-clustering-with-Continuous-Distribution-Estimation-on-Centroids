import torch
import torch.nn as nn
import numpy as np
import collections
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

SEED = 9159
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministics = True
torch.backends.cudnn.benchmark = False

class RBF(nn.Module):
    def __init__(self, sigma=None):
        super(RBF, self).__init__()
        self.sigma = sigma

    def forward(self, X, Y):
        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())

        dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

        # Apply the median heuristic (PyTorch does not give true median)
        if self.sigma is None:
            np_dnorm2 = dnorm2.detach().cpu().numpy()
            h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
            sigma = np.sqrt(h).item()
        else:
            sigma = self.sigma

        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        kxy = (-gamma * dnorm2).exp()  # c x c

        dxkxy = -torch.matmul(kxy, X)  # c x d
        sumkxy = torch.sum(kxy, axis=1).unsqueeze(dim=1)  # c x 1
        dxkxy = (dxkxy + X * sumkxy) * 2 * gamma  # c x d = c x d + c x d * c x 1
        return kxy, dxkxy

class SVGD(nn.Module):
    def __init__(self, flag_cuda, num, dim, r1_overlap=0.5, r2_useless=0.1, v=5, sig=1, stepsize=1e-1, mu_ini= None, base_distribution='Student-1', alpha=0.01):
        super(SVGD, self).__init__()
        self.K = RBF()
        self.base_distribution = base_distribution
        self.num = num
        self.dim = dim
        self.r1 = r1_overlap
        self.r2 = r2_useless
        self.v = v
        self.sig = sig
        self.step_size = stepsize
        self.alpha = alpha
        if mu_ini == None:
            self.mu = torch.nn.init.xavier_uniform_(torch.empty(self.num, self.dim))*10
        else:
            self.mu = mu_ini
        self.historical_grad = torch.tensor(0)
        self.K_XX = torch.tensor(0)
        self.pi = torch.ones(self.num,1) / self.num  # the weight of each centroid
        self.ratio = torch.ones(self.num,1) / self.num  # the ratio of samples assigned to each centroid
        self.normal_prob = None # assignment matrix 
        if flag_cuda:
            self.mu = self.mu.cuda()
            self.historical_grad = self.historical_grad.cuda()
            self.K_XX = self.K_XX.cuda()
            self.pi = self.pi.cuda()
            self.ratio = self.ratio.cuda()

    def log_likelihood(self, Dist, z_dist):
        if self.base_distribution == 'Gaussian':
            ## log coefficients
            ln2piD = torch.log(torch.tensor(2 * np.pi * self.sig)) * self.mu.shape[1]
            ## log likelihood of each sample
            log_components = -0.5 * (ln2piD + 1/self.sig * z_dist)  # n x c

            ## extra coefficient for log gradient 
            neg_grad = 1/self.sig * Dist  # n x c x d
        elif self.base_distribution == 'simple_Gaussian':
            ## log likelihood of each sample
            log_components = -0.5 /self.sig * z_dist  # n x c

            ## extra coefficient for log gradient 
            neg_grad = 1/self.sig * Dist  # n x c x d
        elif self.base_distribution == 'Student-1':
            ## log likelihood of each sample
            log_components = -(torch.log(torch.tensor(np.pi)) + torch.log(1 + z_dist))  # n x c
                       
            ## extra coefficient for log gradient 
            neg_grad = 2 * Dist  / (1 + z_dist.unsqueeze(dim=2)) # n x c x d
        elif self.base_distribution == 'Student-v':
            ## log likelihood of each sample
            log_components = -(self.v + 1)/2 * torch.log(1 + z_dist/self.v)  # n x c

            ## extra coefficient for log gradient 
            neg_grad = (self.v +1) * Dist  / (self.v + z_dist.unsqueeze(dim=2)) # n x c x d
        return log_components, neg_grad

    def log_prob(self, data):
        ## squre distance between sample and centroid
        Dist = torch.unsqueeze(data, 1) - self.mu  # n x c x d
        z_dist = torch.sum(torch.mul(Dist, Dist), 2)  # n x c
        log_components, neg_grad = self.log_likelihood(Dist, z_dist)
        ## log likelihood plus cluster ratio
        if self.pi is None:
            cluster_ratio = torch.tensor(1 / self.mu.shape[0])  # for average clustering ratio
        else:
            cluster_ratio = self.ratio.squeeze().unsqueeze(dim=0)
        log_weighted = log_components + torch.log(cluster_ratio)  # n x c + c x 1
        ## the maximum item of each sample to avoid outflow
        log_shift, _ = log_weighted.max(dim=1)  # n x 1
        ## the normalized probability of each sample over all clusters
        prob = torch.exp(log_weighted - log_shift.unsqueeze(dim=1))  # n x c, n x 1  # likelihood
        self.normal_prob = prob / torch.sum(prob, axis=1).unsqueeze(dim=1)  # n x c, n x 1 assignment propability n x c
        ## the log likelihood
        # logp = torch.log(prob.sum(dim=1)) + log_shift  # n x c, n x 1
        # score = torch.mean(logp, dim=0)

        ## log gradient
        grad = torch.mul(self.normal_prob.unsqueeze(dim=2), neg_grad)  # n x c x 1   * n x c x d
        grad = torch.mean(grad, dim=0)
        # self.ratio = self.normal_prob.sum(dim=0)/self.normal_prob.sum()  # clsuter ratio
        # self.normal_prob = self.normal_prob.detach()
        # self.ratio = self.ratio.detach()
        return grad

    def pre_train_cluster_phi(self, data):
        grad_prob = self.log_prob(data)
        self.K_XX, grad_kxx = self.K(self.mu, self.mu.detach())
        self.K_XX = self.K_XX.detach()
        phi = (self.K_XX.matmul(grad_prob) + self.alpha / self.mu.shape[0]  * grad_kxx) / self.mu.shape[0] 
        return phi

    def cluster_phi(self, data):
        grad_prob = self.log_prob(data)
        self.K_XX, grad_kxx = self.K(self.mu, self.mu.detach())
        self.K_XX = self.K_XX.detach()
        phi = (self.K_XX.matmul(grad_prob) + self.alpha * self.ratio * grad_kxx) * self.ratio
        return phi

    def cluster_step(self, iter, data, flag=False, beta=0.9, eps=1e-6):
        if flag:
            grad = self.pre_train_cluster_phi(data)
        else:
            grad = self.cluster_phi(data)
        # adagrad
        if iter == 0:
            self.historical_grad = self.historical_grad + grad ** 2
        else:
            self.historical_grad = beta * self.historical_grad + (1 - beta) * (grad ** 2)
        adj_grad = torch.div(grad, eps + torch.sqrt(self.historical_grad))
        self.mu = self.mu + self.step_size * adj_grad
        self.mu = self.mu.detach()
        self.historical_grad = self.historical_grad.detach()
        return grad

    def group_similarity(self, data):
        Dist = torch.unsqueeze(data, 1) - self.mu  # n x c x d
        z_dist = torch.sum(torch.mul(Dist, Dist), 2)  # n x c
        gamma = F.softmax(-1 * z_dist, dim=1)
        ratio = gamma.sum(axis=0)/gamma.sum()
        descend_idx = np.argsort(0-ratio.cpu().detach().numpy()) 
        re_Data_id = gamma[:, descend_idx].cpu().detach().numpy()
        group_sim = cos_sim(re_Data_id.T, re_Data_id.T) 
        group_sim = torch.tensor(group_sim, device=self.mu.device).float()
        ratio = ratio.to(self.mu.device)
        return group_sim, descend_idx, ratio

    def shrinkage(self, cos_sim, descend_idx, ratio, Iter=None, M0=None):
        r2 = self.r2 * 1/cos_sim.shape[0]
        ratio = ratio.unsqueeze(dim=1)
        self.ratio = self.ratio + 0.75*ratio
        self.ratio = self.ratio / self.ratio.sum()
        if Iter == None:
            r1 = self.r1
        else:
            r1 = 0.3 + (self.r1-0.3) * np.exp(0.01 * (M0-Iter))   
        print('R1_overlap:{} R2_useless:{}'.format(r1, self.r2)) 
        A, B = torch.where(cos_sim > r1)
        idx = torch.where(A < B)
        if idx[0].shape[0] > 0:
            Id = set(descend_idx[B[idx[0]].cpu().numpy()])
            print('Shinkage {} type-1 sibling centroid and {} remaining'.format(Id.__len__(), cos_sim.shape[0] - Id.__len__()))
            re_id = set(np.arange(self.mu.size(0)))
            re_id = list(re_id.difference(Id))
            self.mu = self.mu[re_id]
            self.historical_grad = self.historical_grad[re_id]
            dict = collections.Counter(descend_idx[B[idx[0]].tolist()])
            N = idx[0].__len__() - 1
            for n in range(idx[0].__len__()):
                self.ratio[descend_idx[A[idx[0][N-n]]]] = self.ratio[descend_idx[A[idx[0][N-n]]]] + self.ratio[descend_idx[B[idx[0][N-n]]]] * 1/dict[descend_idx[B[idx[0][N-n]]].item()]
            self.ratio = self.ratio[re_id]
            return self.mu

        idx = torch.where(self.ratio < r2)
        if idx[0].shape[0] > 0:
            Id = set(idx[0].cpu().numpy())
            print('Shinkage {} useless centroid and {} remaining'.format(Id.__len__(), self.K_XX.shape[0] - Id.__len__()))
            re_id = set(np.arange(self.mu.size(0)))
            re_id = list(re_id.difference(Id))
            self.mu = self.mu[re_id]
            self.historical_grad = self.historical_grad[re_id]
            self.ratio = self.ratio[re_id]
            self.ratio = self.ratio / self.ratio.sum()
            return self.mu