import sys
sys.path.append("../../..")
sys.path.append("..")
import random
import numpy as np
import torch
import os
from utils import build_flags, est_single_causality
from TRDCDL.Module import ENCODER, DNGC_DCV
from STCL.Modules.LoadData.load_data_ADNI import get_dataloader


parser = build_flags()

args = parser.parse_args(args=[])
args.gpu_idx = 0
torch.cuda.set_device(args.gpu_idx)
args.lag = 1
args.sparsity_type = 'log_sum'
args.divergence_type = 'KL'
args.batch_size = 256
args.val_epochs = 2000
args.lr_val = 1e-1
args.val_hidden = 64
args.val_dropout = 0.3

args.est_epochs = 2000
args.lr_est = 1e-2
args.beta_kl = 5e-3
args.beta_sparsity = 2e-1
args.beta_mmd = 1e-3
args.est_hidden = 32
args.est_dropout = 0.3

args.num_nodes = 90
args.threshold = 0.1
args.seed = 80

args.seed = 0
args.root_folder = r'Result_ANDI'
if not os.path.exists(args.root_folder):
    os.makedirs(args.root_folder)

print(args)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_loader = get_dataloader(batch_size = args.batch_size, parent=3)

single_data_loader = get_dataloader(batch_size = 1, parent=3)
ENCODER(data_loader=data_loader, n_in=args.dims, n_hid=args.val_hidden, num_node=args.num_nodes,
            num_epoch=args.val_epochs, lr=args.lr_val, weight_decay=args.weight_decay,
            save_folder=args.root_folder, do_prob=args.val_dropout)

DNGC_DCV(data_loader, n_in=args.dims, n_hid=args.est_hidden, num_node=args.num_nodes, num_epoch=args.est_epochs,
            lr=args.lr_est, weight_decay=args.weight_decay, save_folder=args.root_folder, n_in_val=args.dims,
            n_hid_val=args.val_hidden, sparsity_type=args.sparsity_type, divergence_type=args.divergence_type,
            do_prob=args.est_dropout, beta_sparsity=args.beta_sparsity, beta_kl=args.beta_kl, beta_mmd=args.beta_mmd)
folder_name = "ECNs_results"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
for data, id, group in single_data_loader:
    adj_gca = est_single_causality(data, args.dims, args.est_hidden, args.num_nodes, args.root_folder)
    out = np.array(adj_gca.cpu())
    folder = "ECNs_results/" + id[0] + "_" + group[0] + ".npy"
    np.save(folder, out)

