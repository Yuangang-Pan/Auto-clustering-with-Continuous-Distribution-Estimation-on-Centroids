import math
import torch
import numpy as np
from numpy import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as PathEffects
from sklearn import mixture
from sklearn.neighbors import KernelDensity
import time
## add path
import sys
import psutil
import os

add = ''
path = add + '/SVGD/'
file = add + '/Results/Ablation_M4G/'
sys.path.append(path)

import SVGD as Stein

flag_cuda = torch.cuda.is_available()
SEED = 2022
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministics = True
torch.backends.cudnn.benchmark = False

matplotlib.rcParams['axes.edgecolor'] = '#808080'
matplotlib.rcParams['xtick.color'] = '#808080'
matplotlib.rcParams['ytick.color'] = '#808080'
matplotlib.rcParams.update({'font.size': 8})

center_color = '#000000'
data_color = np.array([[0.86, 0.3712, 0.34]])
grid_color = '#dfdfdf'

def Simulation(type='one_G', flag = 'None', List_num='', List_mu='', List_cov=''):
    X = []
    Y = []
    Temp_num = []
    Temp_mu = []
    Temp_cov = []
    if type == 'skrinage':
        T = 6
        for i in range(T):
            if flag == 'None':
                Num = np.random.randint(300, 800)
                mu = 10 * np.random.random(2) - 5  
                temp_cov = np.random.rand(2,2)
                cov = (temp_cov + temp_cov.T)/2 + 0.5*np.eye(2)
                Temp_num.append(Num)
                Temp_mu.append(mu)
                Temp_cov.append(cov)
            else:
                Num = List_num[i] * flag
                mu = List_mu[i]
                cov = List_cov[i]
                
            temp_x = np.random.multivariate_normal(mu, cov, Num)
            X.append(temp_x)
            Y.append(np.ones(temp_x.shape[0], dtype=np.int)*i)
    if type == 'one_G':
        mu = np.array([0,0])
        cov = np.diag([1, 1])
        temp_x = np.random.multivariate_normal(mu, cov, Num)
        X.append(temp_x)
    if type == 'two_G':
        mu1 = np.array([-5.0, 0.0])
        mu2 = np.array([5.0, 0.0])
        cov = np.diag([1, 1])

        temp_1 = np.random.multivariate_normal(mu1, cov, Num)
        X.append(temp_1)
        temp_2 = np.random.multivariate_normal(mu2, cov, Num)
        X.append(temp_2)
    if type == 'six_G':
        for i in range(1, 7):
            mu = 5.0 * np.array([np.sin(i * math.pi / 3.0), np.cos(i * math.pi / 3.0)])
            cov = np.diag([1, 1])
            temp_x = np.random.multivariate_normal(mu, cov, Num)
            X.append(temp_x)
    X = np.concatenate(X, 0)
    Y = np.concatenate(Y, 0)
    if flag == 'None':
        return X, Y, Temp_num, Temp_mu, Temp_cov
    else:
        return X, Y

def KDE(Data):
    model = KernelDensity(bandwidth=0.2)
    model.fit(Data)
    
    XX, YY = np.meshgrid(np.arange(-7, 7, 0.01), np.arange(-7, 7, 0.01))
    xy = np.vstack([YY.ravel(), XX.ravel()]).T
    zz = np.exp(model.score_samples(xy))    
    ZZ = zz.reshape(XX.shape)
    return XX, YY, ZZ

def measure(y_true, y_pred):
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    from scipy.optimize import linear_sum_assignment

    NMI_fun = normalized_mutual_info_score
    ARI_fun = adjusted_rand_score

    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max([y_pred.max(), y_true.max()]) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    Acc = w[row_ind, col_ind].sum() / y_pred.size
    nmi = NMI_fun(y_true, y_pred)
    ari = ARI_fun(y_true, y_pred)
    return Acc, nmi, ari

def scatter(x, colors):
    n_color = colors.max() + 1
    palette = np.array(sns.color_palette("hls", n_color))
    f = plt.figure()
    ax = plt.subplot()
    ax.set_axisbelow(True)
    ax.grid(True)
    
    for i in range(n_color):
        positions = np.where(colors == i)
        ax.scatter(x[positions[0], 0], x[positions[0], 1], s=5, alpha=1, marker='.', c=palette[colors[positions[0]].astype(np.int)], label='{}'.format(i))
        
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=10)
        txt.set_path_effects([PathEffects.Stroke(linewidth=2, foreground="w"), PathEffects.Normal()])        
    
    name = file + 'M4G_data_{}.png'.format(x.size)  
    plt.savefig(name) 
    plt.close()

def dp_means(data, Lambda, max_iters=100, tolerance=10e-3):

    n = len(data)
    k = 1
    assignments = ones(n).astype(int)
    mu = []
    for d in range(data.ndim):
        mu.append(array((mean(data[:, d]),)))
    is_converged = False
    n_iters = 0
    ss_old = float('inf')
    ss_new = float('inf')
    while not is_converged and n_iters < max_iters:
        n_iters += 1
        for i in range(n):
            distances = repeat(None, k)
            for j in range(k):
                distances[j] = sum((data[i, d] - mu[d][j]) ** 2  for d in range(data.ndim))
            if min(distances) > Lambda:
                k += 1
                assignments[i] = k-1
                for d in range(data.ndim):                    
                    mu[d] = append(mu[d], data[i, d])
            else:
                assignments[i] = argmin(distances)
        for j in range(k):
            if len(where(assignments == j)) > 0:
                for d in range(data.ndim):
                    mu[d][j] = mean(data[assignments == j, d])
        ss_new = 0      
        for i in range(n):
            ss_new <- ss_new + sum((data[i, d] - mu[d][assignments[i]]) ** 2 for d in range(data.ndim))
        ss_change = ss_old - ss_new
        is_converged = ss_change < tolerance  
    return {'centers': column_stack([mu[d] for d in range(data.ndim)]), 'assignments': assignments, 'k': k, 'n_iters': n_iters}


def dp_means_fast(data, Lambda, max_iters=100, tolerance=10e-3):

    n = len(data)
    k = 1
    assignments = zeros(n).astype(int)
    mu = []
    ndim = data.shape[1]
    mu = np.expand_dims(np.mean(data, 0), 1)  # ndim*1
    is_converged = False
    n_iters = 0
    ss_old = float('inf')
    ss_new = float('inf')
    while not is_converged and n_iters < max_iters:
        n_iters += 1
        # print(n_iters)
        for i in range(n):         
            # add new mus and decide assigments 
            distances = sum((expand_dims(data[i], 1)-mu)**2, 0)
            if min(distances) > Lambda:
                k += 1
                assignments[i] = k-1
                mu = concatenate([mu, expand_dims(data[i], 1)], 1)
            else:
                assignments[i] = argmin(distances)
        # update mu
        mu = [np.mean(data[assignments==j], 0) for j in range(k)] 
        mu = np.stack(mu, 1) # n*k
        # check convergence
        ss_new = np.sum((data-mu[:, assignments].T)**2) 
        ss_change = ss_old - ss_new
        is_converged = ss_change < tolerance 
        ss_old = ss_new 
    return {'centers': column_stack([mu[d] for d in range(ndim)]), 'assignments': assignments, 'k': k, 'n_iters': n_iters}


### code start from here
Vanila_data_generation = '' ## Whether to regenerate data. To ensure consistency, I have already saved all the data.
visualization = ''  ## Whether to visualize, mainly to observe the data distribution
data_type = 'skrinage' ## Our data generation type
PA = 100 ## Data amplification factor
data_name = file + 'M4G_data_{}'.format(PA) ## Each dataset uses the amplification factor as a suffix

if Vanila_data_generation:
    if Vanila_data_generation == 'True': ## To keep the data distribution consistent, for the Vanilla dataset, we need to save its mean, covariance, and sample size for each cluster
        data, labels, Temp_num, Temp_mu, Temp_cov = Simulation(data_type)
        np.savez(file + 'hyper_parameter.npz', NUM=Temp_num, MU=Temp_mu, COV=Temp_cov)
    else:
        hyper_para = np.load(file + 'hyper_parameter.npz') ## For the amplified dataset, we need to load the vanilla hyperparameters to generate data with the same distribution but larger sample size
        data, labels = Simulation(data_type, PA, hyper_para['NUM'], hyper_para['MU'], hyper_para['COV'])
    scatter(data, labels) # from TNSE we find cluster 1-5 collapse and 3-4 collapse
    labels[labels == 4] = 3 ## Actually, we generated 6 overlapping Gaussians, but only 4 are shown. According to the visualization, clusters 3-4 overlap into one, and clusters 1-5 overlap into one.
    labels[labels == 5] = 1
    
    if Vanila_data_generation == 'True':
        np.savez(data_name + '.npz', X=data, Y=labels)
    else:
        np.savez(data_name + '_expand.npz', X=data, Y=labels)
## load M4G data 
if PA == 1:
    M4G = np.load(file + 'M4G_data.npz')
else:
    M4G = np.load(data_name + '_expand.npz')
data = M4G['X']
data= np.float32(data)
labels = M4G['Y']
print('data size: ', data.shape, labels.shape)

fig = plt.figure(figsize=(10, 2))
## We have 5 nonparametric methods, execute one each time
Baseline = ['DP-GMM', 'DP-means', 'HAC', 'DBSCAN', 'ACT']
method = Baseline[4]
print('method:', method)
memory_usage = []
"""
Plot the Data density via the KDE
"""
if visualization == 'True': ## Use kernel density estimation (KDE) to visualize the data
    ax = fig.add_subplot(1, 4, 1)
    ax.set_axisbelow(True)
    ax.grid(True)
    X, Y, Z = KDE(data)
    ax.contourf(Y, X, Z, levels = np.linspace(0, Z.max(), 25), cmap = 'Reds')
    ax.xaxis.set_ticks([-5.0, -2.5, 0.0, 2.5, 5.0])  
    ax.yaxis.set_ticks([-5.0, -2.5, 0.0, 2.5, 5.0])  
    fig.tight_layout()
    plt.savefig(data_name + '_data_kde.png', bbox_inches='tight')     

    plt.clf()
    ax = fig.add_subplot(1, 4, 1)
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.scatter(data[:, 0], data[:, 1], s=1, alpha=1, marker='.', c=np.array([[0.86, 0.3712, 0.34]]))
    fig.tight_layout()
    plt.savefig(data_name + '_data_scatter.png', bbox_inches='tight') 

if method == 'DP-GMM':
    """
    Fit a Dirichlet process Gaussian mixture (DP-GMM) using 50 components
    """
    start = time.perf_counter()
    dpgmm = mixture.BayesianGaussianMixture(n_components=50, covariance_type="full", max_iter=1000, random_state=2022).fit(data)
    process = psutil.Process(os.getpid())
    current_process_memory = process.memory_info().rss 
    allocated_memory = current_process_memory / (1024 ** 2)
    memory_usage.append(allocated_memory)
    print(f"DP-GMM - Allocated fitting Memory: {allocated_memory:.2f} MB")

    y_preds = dpgmm.predict(data)
    process = psutil.Process(os.getpid())
    current_process_memory = process.memory_info().rss 
    allocated_memory = current_process_memory / (1024 ** 2)
    memory_usage.append(allocated_memory)
    print(f"DP-GMM - Allocated predeication Memory: {allocated_memory:.2f} MB")

    end = time.perf_counter()
    mu = dpgmm.means_
    n_components = dpgmm.n_components
    acc, nmi, ari = measure(labels, y_preds)
    
    pi = [np.sum(y_preds==i)/y_preds.shape[0] for i in range(n_components)]
    idx = np.array(pi) > 0.001
    num_cluster = np.sum(idx)
    print('DP-GMM #cluster: {}'.format(num_cluster)) 
    print('Time:{:.4f} \tACC:{:.4f} \tNMI:{:.4f} \tARI:{:.4f}'.format(end-start, acc, nmi, ari))
    
    n_color = y_preds.max()+2
    palette = np.array(sns.color_palette("hls", n_color))
    plt.clf()
    ax = fig.add_subplot(1, 4, 1)
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.scatter(data[:, 0], data[:, 1], s=1, alpha=1, marker='.', c=palette[y_preds])
    size = 100 * np.array(pi)
    print(size.shape, mu.shape)
    ax.scatter(mu[idx, 0], mu[idx, 1], s=size[idx], alpha=1, marker='*', c=center_color)
    fig.tight_layout()
    name = file + '{}_k{}.png'.format('DP-GMM', num_cluster) 
    plt.savefig(name, bbox_inches='tight')
    plt.close()
elif method == 'DP-means':
    """
    Fit a Dirichlet process k-means (DP-means) using 50 components
    """
    start = time.perf_counter()
    results = dp_means_fast(data, 50)
    process = psutil.Process(os.getpid())
    current_process_memory = process.memory_info().rss 
    allocated_memory = current_process_memory / (1024 ** 2)
    memory_usage.append(allocated_memory)
    print(f"DP-means - Allocated fitting Memory: {allocated_memory:.2f} MB")

    y_preds = results['assignments']
    process = psutil.Process(os.getpid())
    current_process_memory = process.memory_info().rss 
    allocated_memory = current_process_memory / (1024 ** 2)
    memory_usage.append(allocated_memory)
    print(f"DP-means - Allocated predeication Memory: {allocated_memory:.2f} MB")

    end = time.perf_counter()
    mu = results['centers']
    acc, nmi, ari = measure(labels, y_preds)
    pi = [np.sum(y_preds==i)/y_preds.shape[0] for i in range(y_preds.max()+1)]
    
    num_cluster = np.sum(np.array(pi) > 0.001)
    pi = np.array(pi)[np.array(pi) > 0.001]
    print('DP-means #cluster: {}'.format(num_cluster)) 
    print('Time:{:.4f} \tACC:{:.4f} \tNMI:{:.4f} \tARI:{:.4f}'.format(end-start, acc, nmi, ari))
    
    n_color = y_preds.max()+2
    palette = np.array(sns.color_palette("hls", n_color))
    plt.clf()
    ax = fig.add_subplot(1, 4, 1)
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.scatter(data[:, 0], data[:, 1], s=1, alpha=1, marker='.', c=palette[y_preds+1])
    size = 100 * np.array(pi)
    ax.scatter(mu[:, 0], mu[:, 1], s=size, alpha=1, marker='*', c=center_color)
    fig.tight_layout()
    name = file + '{}_k{}.png'.format('DP-means', num_cluster) 
    plt.savefig(name, bbox_inches='tight')
    plt.close()
elif method == 'HAC':
    """"
    Hierarchical clustering
    """
    from sklearn.cluster import AgglomerativeClustering
    start = time.perf_counter()
    ac = AgglomerativeClustering(n_clusters=None, distance_threshold=60).fit(data)
    process = psutil.Process(os.getpid())
    current_process_memory = process.memory_info().rss 
    allocated_memory = current_process_memory / (1024 ** 2)
    memory_usage.append(allocated_memory)
    print(f"HAC - Allocated inference Memory: {allocated_memory:.2f} MB")

    end = time.perf_counter()
    y_preds = ac.labels_
    acc, nmi, ari = measure(labels, y_preds)
    num_cluster = y_preds.max()+1
    print('HAC #cluster: {}'.format(num_cluster)) 
    print('Time:{:.4f} \tACC:{:.4f} \tNMI:{:.4f} \tARI:{:.4f}'.format(end-start, acc, nmi, ari))
    
    n_color = y_preds.max()+2
    palette = np.array(sns.color_palette("hls", n_color))
    plt.clf()
    ax = fig.add_subplot(1, 4, 1)
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.scatter(data[:, 0], data[:, 1], s=1, alpha=1, marker='.', c=palette[y_preds+1])
    fig.tight_layout()
    name = file + '{}_k{}.png'.format('HAC', num_cluster) 
    plt.savefig(name, bbox_inches='tight')
    plt.close()
elif method == 'DBSCAN':
    """
    DBSCAN
    """
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    start = time.perf_counter()
    norm_data = StandardScaler().fit_transform(data)
    db = DBSCAN(eps=0.2, min_samples=15).fit(norm_data) # min_samples=12-15
    process = psutil.Process(os.getpid())
    current_process_memory = process.memory_info().rss 
    allocated_memory = current_process_memory / (1024 ** 2)
    memory_usage.append(allocated_memory)
    print(f"DBSCAN - Allocated inference Memory: {allocated_memory:.2f} MB")

    end = time.perf_counter()
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    y_preds = db.labels_
    acc, nmi, ari = measure(labels, y_preds)
    print('DBSCAN #cluster: {} \t#noise points:{}'.format(y_preds.max()+1, np.sum(y_preds==-1))) 
    print('Time:{:.4f} \tACC:{:.4f} \tNMI:{:.4f} \tARI:{:.4f}'.format(end-start, acc, nmi, ari))

    noise_data = data[y_preds==-1]
    noise_y_preds = y_preds[y_preds==-1]
    cluster_data = data[y_preds>-1]
    cluster_y_preds = y_preds[y_preds>-1]
    ## plot tsne with noise samples
    n_color = y_preds.max() + 2
    palette = np.array(sns.color_palette("hls", n_color))
    plt.clf()
    ax = fig.add_subplot(1, 4, 1)
    ax.set_axisbelow(True)
    ax.grid(True)
    fig.tight_layout()
    ax.scatter(noise_data[:, 0], noise_data[:, 1], s=1, alpha=1, marker='s', c='grey')
    ax.scatter(cluster_data[:, 0], cluster_data[:, 1], s=1, alpha=1, marker='.', c=palette[cluster_y_preds])
    name = file + '{}_k{}.png'.format('DBSCAN', y_preds.max()+1) 
    plt.savefig(name, bbox_inches='tight') 
    plt.close()
else:
    """
    Our ACT
    """
    device = 0
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)

    from sklearn.mixture import GaussianMixture
    if flag_cuda:
        data = torch.tensor(data, dtype=torch.float).cuda(device)
        labels = torch.from_numpy(labels).cuda(device)
    # ## for fiting 50 centroids
    
    num = 50
    dim = 2
    Gap = 10
    
    gmm = GaussianMixture(n_components=num, covariance_type='spherical', random_state=SEED).fit(data.cpu().numpy())
    mu = torch.tensor(gmm.means_, dtype=torch.float).to(device)
    
    start = time.perf_counter()
    svgd = Stein.SVGD(flag_cuda, num, dim, mu_ini=mu)
    mu0 = svgd.mu.cpu().numpy()
    Step = 0
    
    for iter in range(1000):
        if iter <= 200:
            svgd.cluster_step(iter, data)

            # allocated_memory = torch.cuda.memory_allocated(data.device) / (1024 ** 2)
            
            free, total = torch.cuda.mem_get_info(device) 
            allocated_memory = (total - free) / (1024 ** 2)
    
            if iter == 200:
                mu_SVGD = svgd.mu.cpu().numpy()
        else:
            flag = iter > 200
            svgd.cluster_step(iter, data, flag)
            if iter % Gap ==0:
                group_sim, descend_idx, ratio = svgd.group_similarity(data)
                Flag = svgd.shrinkage(group_sim, descend_idx, ratio)
                if Flag == 'True':
                    Step = 0
                else: 
                    Step = Step + Gap

            # allocated_memory = torch.cuda.memory_allocated(data.device) / (1024 ** 2)
            free, total = torch.cuda.mem_get_info(device) 
            allocated_memory = (total - free) / (1024 ** 2)
        if Step >=100:
            break
        memory_usage.append(allocated_memory)
        print(f"Iter {iter+1} - : {allocated_memory:.2f} MB")
    end = time.perf_counter()
    Dist = torch.unsqueeze(data, 1) - svgd.mu  # n x c x d
    z_dist = torch.sum(torch.mul(Dist, Dist), 2)  
    max_gamma, y_preds = z_dist.min(dim=1)
    
    data = data.cpu().numpy()
    labels = labels.cpu().numpy()
    mu_ACT = svgd.mu.cpu().numpy()
    y_preds = y_preds.cpu().numpy()
    acc, nmi, ari = measure(labels, y_preds)
    print('ACT #cluster: {}'.format(y_preds.max()+1)) 
    print('Time:{:.4f} \tACC:{:.4f} \tNMI:{:.4f} \tARI:{:.4f}'.format(end-start, acc, nmi, ari))
np.save(data_name + '_' + method + '_memory.npy', np.array(memory_usage))

    
