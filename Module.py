from Model import *
from utils import loss_sparsity, loss_divergence, set_est_graph_kernel, loss_mmd
import os
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
from tqdm import tqdm

def ENCODER_i(data_loader, idx, n_in, n_hid, num_node, do_prob, num_epoch, lr, weight_decay, log, val_file):
    net = encoder_TRDCDL(n_in, n_hid, num_node, do_prob)
    net = net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.3)
    loss_val = nn.MSELoss()
    best_loss = np.inf
    for epoch in range(num_epoch):
        scheduler.step()
        t = time.time()
        Loss = []
        mse_loss = []
        for data, id, group in data_loader:

            data = data.unsqueeze(3)
            data = data.cuda()
            target = data[:, 1:, idx,:]
            optimizer.zero_grad()
            data_hat = net(data[:, :-1, :, :])
            mse = loss_val(data_hat, target)
            loss = mse
            loss.backward()
            optimizer.step()
            mse_loss.append(mse.item())
            Loss.append(loss.item())

        if np.mean(mse_loss) < best_loss:
            best_loss = np.mean(mse_loss)
            torch.save(net.state_dict(), val_file)

        log.flush()




def ENCODER(data_loader, n_in, n_hid, num_node, num_epoch, lr, weight_decay, save_folder, do_prob=0.):

    log_file = os.path.join(save_folder, 'log_val.txt')

    log = open(log_file, 'w')
    print('Begin training VALNet')
    for idx in tqdm(range(num_node)):
        val_file = 'VALNet' + str(idx) + '.pt'
        val_file = os.path.join(save_folder, val_file)
        ENCODER_i(data_loader, idx, n_in, n_hid, num_node, do_prob, num_epoch, lr, weight_decay, log, val_file)
    log.close()

def DNGC_DCV_i(data_loader, idx, n_in, n_hid, num_node, do_prob, graph_kernel, num_epoch, lr, weight_decay,
                     sparsity_type, divergence_type,
                     beta_sparsity, beta_kl, beta_mmd, beta_prior, log, est_file, n_in_val, n_hid_val, val_file):
    val_net = encoder_TRDCDL(n_in_val, n_hid_val, num_node)
    val_net.load_state_dict(torch.load(val_file))
    val_net = val_net.cuda()
    val_net.eval()

    # NOTE 逆序，将得到的逆序因果阵先验转置
    graph_kernel1 = graph_kernel.permute(1,0)
    this_kernel = graph_kernel1[:,idx].cuda()

    # # NOTE lorenz逆序，将得到的逆序因果阵先验转置
    # graph_kernel1 = graph_kernel
    # this_kernel = graph_kernel1[:,idx].cuda()

    est_net = DCV(graph_kernel1, n_in, n_hid, 1, do_prob)
    est_net = est_net.cuda()
    optimizer = optim.Adam(est_net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.3)
    loss_mse = nn.MSELoss()
    loss_ce = nn.CrossEntropyLoss()
    best_loss = np.inf

    for epoch in range(num_epoch):
        scheduler.step()
        t = time.time()
        Loss = []
        MSE_loss = []
        SPA_loss = []
        KL_loss = []
        MMD_loss = []
        for data, id, group in data_loader:

            optimizer.zero_grad()
            data = data.unsqueeze(3)
            data = data.cuda()
            # NOTE 逆序1 从时间轴上反转数据
            data = torch.flip(data, dims=[1])

            u, v, mask = est_net(data[:, :-1, :, :])
            # NOTE 逆序 利用正向数据生成该结点对应的mask，反转时间轴，得到逆序mask
            this_kernel0 = this_kernel.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            C_prior0 = torch.repeat_interleave(this_kernel0, mask.size(0),dim=0)
            C_prior = torch.repeat_interleave(C_prior0, mask.size(2),dim=2)
            mask = torch.flip(mask, dims=[2])
            data0 = torch.flip(data, dims=[1])
            
            inputs = mask_inputs(mask.permute(0,2,1,3), data0[:, :-1, :, :])
            pred = val_net(inputs)
            target = data0[:, 1:, idx, :]
            mse_loss = loss_mse(pred, target)
            spa_loss = beta_sparsity * loss_sparsity(mask, sparsity_type)
            z_loss = beta_kl*(loss_divergence(u, divergence_type='entropy')+loss_divergence(v, divergence_type='entropy'))
            C_prior_loss = beta_prior*loss_ce(mask, C_prior)
            loss = mse_loss + spa_loss + z_loss + C_prior_loss
            loss.backward()
            optimizer.step()
            
            Loss.append(loss.item())
            MSE_loss.append(mse_loss.item())
            SPA_loss.append(spa_loss.item())
            KL_loss.append(z_loss.item())
            MMD_loss.append(C_prior_loss.item())

        if np.mean(Loss) < best_loss:
            best_loss = np.mean(Loss)
            torch.save(est_net.state_dict(), est_file)
        log.flush()


def DNGC_DCV(data_loader, n_in, n_hid, num_node, num_epoch, lr, weight_decay, save_folder, n_in_val, n_hid_val,
              sparsity_type='l2', divergence_type='entropy', do_prob=0., beta_sparsity=1, beta_kl=0.1, beta_mmd=1, beta_prior=0.1):
    log_file = os.path.join(save_folder, 'log_est.txt')
    log = open(log_file, 'w')
    graph_kernel = set_est_graph_kernel(save_folder, n_in_val, n_hid_val, num_node)
    print('Begin training ESTNet')
    for idx in tqdm(range(num_node)):
        val_file = 'VALNet' + str(idx) + '.pt'
        val_file = os.path.join(save_folder, val_file)
        est_file = 'ESTNet' + str(idx) + '.pt'
        est_file = os.path.join(save_folder, est_file)
        DNGC_DCV_i(data_loader, idx, n_in, n_hid, num_node, do_prob, graph_kernel, num_epoch, lr, weight_decay,
                         sparsity_type, divergence_type,
                         beta_sparsity, beta_kl, beta_mmd, beta_prior, log, est_file, n_in_val, n_hid_val, val_file)
    log.close()
