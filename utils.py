import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from Model import DCV, encoder_TRDCDL
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as prfs


def build_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-idx', type=int, default=0, help='set the gpu')
    parser.add_argument('--seed', type=int, default=620, help='Random seed.')
    parser.add_argument('--num-nodes', type=int, default=5,
                        help='Number of nodes in simulation.')
    parser.add_argument('--dims', type=int, default=1,
                        help='The number of input dimensions.')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='The threshold of evaluating causality.')
    parser.add_argument('--time-length', type=int, default=1000,
                        help='The length of time series.')
    parser.add_argument('--val-epochs', type=int, default=300,
                        help='Number of epochs to train the val net.')
    parser.add_argument('--est-epochs', type=int, default=500,
                        help='Number of epochs to train the est net.')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Number of samples per batch.')
    parser.add_argument('--lr_val', type=float, default=1e-2,
                        help='Initial learning rate.')
    parser.add_argument('--lr_est', type=float, default=1e-3,
                        help='Initial learning rate.')
    parser.add_argument('--weight-decay', type=float, default=1e-3,
                        help='Weight decay.')
    parser.add_argument('--sparsity-type', type=str, default='l2',
                        help='The type of sparsity loss.')
    parser.add_argument('--beta-sparsity', type=float, default=1,
                        help='The Weight of sparsity loss.')
    parser.add_argument('--beta-kl', type=float, default=1e-1,
                        help='The Weight of KL loss.')
    parser.add_argument('--beta-mmd', type=float, default=1,
                        help='The Weight of MMD loss.')
    parser.add_argument('--est-hidden', type=int, default=15,
                        help='Number of hidden units.')
    parser.add_argument('--val-hidden', type=int, default=15,
                        help='Number of hidden units.')
    parser.add_argument('--est-dropout', type=float, default=0.3,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--val-dropout', type=float, default=0.3,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--root-folder', type=str, default='logs',
                        help='Where to save the trained model, leave empty to not save anything.')
    return parser


def time_split(T, step=10):
    start = 0
    end = step
    samples = []
    while end <= T.shape[1]:
        samples.append(T[:, start:end, :])
        start += 1
        end += 1
    return samples


def get_val_graph_kernel(root_folder, n_in, n_hid, num_node):
    graph_kernel = []
    for idx in range(num_node):
        val_file = 'VALNet' + str(idx) + '.pt'
        val_file = os.path.join(root_folder, val_file)
        val_net = encoder_TRDCDL(n_in, n_hid, num_node)
        val_net.load_state_dict(torch.load(val_file))
        graph_kernel.append(val_net.graph_kernel[0])
    return graph_kernel


def set_est_graph_kernel(save_folder, n_in, n_hid_val, num_node):
    graph_kernel = get_val_graph_kernel(save_folder, n_in, n_hid_val, num_node)
    graph_kernel = torch.cat([temp.unsqueeze(0) for temp in graph_kernel], dim=0)
    graph_kernel = graph_kernel.clone().detach()
    return graph_kernel


def get_est_graph_kernel(root_folder, n_in, n_hid, num_node, ):
    graph_kernel = []
    init_graph_kernel = torch.eye(num_node)
    for idx in range(num_node):
        est_file = 'ESTNet' + str(idx) + '.pt'
        est_file = os.path.join(root_folder, est_file)
        est_net = DCV(init_graph_kernel, n_in, n_hid)
        est_net.load_state_dict(torch.load(est_file))
        graph_kernel.append(est_net.graph_kernel)
    return graph_kernel


def evaluate_result(causality_true, causality_pred, threshold):
    causality_pred[causality_pred > 1] = 1
    causality_true = np.abs(causality_true).flatten()
    causality_pred = np.abs(causality_pred).flatten()
    roc_auc = roc_auc_score(causality_true, causality_pred)
    fpr, tpr, _ = roc_curve(causality_true, causality_pred)
    precision_curve, recall_curve, _ = precision_recall_curve(causality_true, causality_pred)
    pr_auc = auc(recall_curve, precision_curve)

    causality_pred[causality_pred > threshold] = 1
    causality_pred[causality_pred <= threshold] = 0
    precision, recall, F1, _ = prfs(causality_true, causality_pred)
    accuracy = accuracy_score(causality_true, causality_pred)

    evaluation = {'accuracy': accuracy, 'precision': precision[1], 'recall': recall[1], 'F1': F1[1],
                  'ROC_AUC': roc_auc, 'PR_AUC': pr_auc}
    plot = {'FPR': fpr, 'TPR': tpr, 'PC': precision_curve, 'RC': recall_curve}
    return evaluation, plot


def count_accuracy(B_true, B_est):
    if (B_est == -1).any():
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')
    else:
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')
    d = B_true.shape[0]
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    acc = (B_true == B_est).mean()
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'acc': acc, 'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}


def summary_result(data):
    idx = 0
    for temp in data:
        temp = pd.DataFrame(temp, index=[idx])
        if idx == 0:
            result = temp
        else:
            result = pd.concat([result, temp])
        idx += 1
    result = result.agg([np.mean, np.std])
    mean = result.loc['mean'].values
    std = result.loc['std'].values
    return mean, std


def est_causality(data_loader, n_in, n_hid, num_node, save_folder):
    causality_matrix = []
    init_graph_kernel = torch.eye(num_node)
    for data, id, group in data_loader:
        data = data.unsqueeze(3)
        data = data.cuda()
        for idx in range(num_node):
            est_file = 'ESTNet' + str(idx) + '.pt'
            est_file = os.path.join(save_folder, est_file)
            est_net = DCV(init_graph_kernel, n_in, n_hid)
            est_net = est_net.cuda()
            est_net.load_state_dict(torch.load(est_file))
            est_net.eval()
            u, v, matrix = est_net(data)
            matrix = matrix.squeeze()
            causality_matrix.append(matrix)

    causality_matrix = torch.stack(causality_matrix, dim=1)
    return causality_matrix

def est_single_causality(data, n_in, n_hid, num_node, save_folder):
    causality_matrix = []
    init_graph_kernel = torch.eye(num_node)

    data = data.unsqueeze(3)
    data = data.cuda()
    for idx in range(num_node):
        est_file = 'ESTNet' + str(idx) + '.pt'
        est_file = os.path.join(save_folder, est_file)
        est_net = DCV(init_graph_kernel, n_in, n_hid)
        est_net = est_net.cuda()
        est_net.load_state_dict(torch.load(est_file))
        est_net.eval()
        u, v, matrix = est_net(data)
        matrix = matrix.squeeze()
        causality_matrix.append(matrix)
    causality_matrix = torch.stack(causality_matrix, dim=1)
    return causality_matrix

def est_single_causality_dist(data, n_in, n_hid, num_node, save_folder):
    causality_matrix = []
    x3_matrix = []
    init_graph_kernel = torch.eye(num_node)

    data = data.unsqueeze(3)
    data = data.cuda()
    for idx in range(num_node):
        est_file = 'ESTNet' + str(idx) + '.pt'
        est_file = os.path.join(save_folder, est_file)
        est_net = DCV(init_graph_kernel, n_in, n_hid)
        est_net = est_net.cuda()
        est_net.load_state_dict(torch.load(est_file))
        est_net.eval()
        u, v, matrix = est_net(data)
        x3 = est_net.x3
        x3_temp = x3.squeeze()
        matrix = matrix.squeeze()
        causality_matrix.append(matrix)
        x3_matrix.append(x3_temp)

    causality_matrix = torch.stack(causality_matrix, dim=1)
    x3_matrix = torch.stack(x3_matrix, dim=1)
    return causality_matrix, x3_matrix


def kl_divergence(x, target):
    epsilon = 1e-6
    return torch.mean(x * torch.log((x+epsilon)/target))


def loss_sparsity(inputs, sparsity_type='l2', epsilon=1e-4):
    if sparsity_type == 'l1':
        return torch.mean(torch.abs(inputs))
    elif sparsity_type == 'log_sum':
        return torch.mean(torch.log(torch.abs(inputs) / epsilon + 1))
    else:
        return torch.mean(inputs ** 2)


def loss_divergence(inputs, divergence_type='entropy', rho=0.1):
    epsilon = 1e-6
    inputs = torch.abs(inputs)
    inputs = inputs.squeeze(dim=3).mean(dim=2).mean(dim=0)
    if divergence_type == 'entropy':
        return torch.mean(inputs * torch.log(inputs + epsilon))
    elif divergence_type == 'JS':
        m = (rho + inputs) / 2
        return kl_divergence(inputs, m) / 2 + kl_divergence(rho, m) / 2
    else:
        return kl_divergence(inputs, rho)


def loss_mmd(x, y, idx, gamma=1):
    loss1 = torch.exp(-1 * gamma * (x - torch.repeat_interleave(x[:, idx:idx + 1, :, :], x.size(1), dim=1)) ** 2)
    loss2 = torch.exp(-1 * gamma * (x - torch.repeat_interleave(y.unsqueeze(1), x.size(1), dim=1)) ** 2)
    loss = torch.abs(torch.mean(loss1) - torch.mean(loss2))
    return loss

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.keys = list(data.keys())
        self.values = list(data.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        key = self.keys[idx]
        value = self.values[idx]
        tensor_value = torch.tensor(value, dtype=torch.float32)
        return key, tensor_value
