import torch
import pandas as pd
from network import Network
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import ContrastiveLoss
from dataloader import load_data
import torch.nn.functional as F
from sklearn.cluster import KMeans
from utils.utils import build_affinity_matrix,graph_fusion
import csv
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Caltech-2V
# Caltech-3V
# Caltech-4V
# Caltech-5V

Dataname = 'Caltech-5V'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--learning_rate", default=0.0003) 
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--pre_epochs", default=200)
parser.add_argument("--con_epochs", default=200)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
parser.add_argument('--neighbor_num', default=3)
parser.add_argument("--temperature", default=1)
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


if args.dataset == "Caltech-2V":
    args.con_epochs = 110 
    seed = 200
if args.dataset == "Caltech-3V":
    args.con_epochs = 110 
    seed = 30
if args.dataset == "Caltech-4V":
    args.con_epochs = 110 
    seed = 100
if args.dataset == "Caltech-5V":
    args.con_epochs = 120 
    seed = 1000000

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(seed)
dataset, dims, view, data_size, class_num = load_data(args.dataset)
data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, zs, _, H = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))

    
def contrastive_train(epoch):
    tot_loss = 0.
    mse = torch.nn.MSELoss()
    
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, zs, Hs, H= model(xs)
        
        adj_list = [build_affinity_matrix(Hs[v], args.neighbor_num, device) for v in range(view)]
        adj_list_new = []
        γ = 2  # 0 1 2 -1 -2
        for i in range(len(adj_list)):
            adj = adj_list[i]
            adj = adj.to(device)
            values = adj._values()
            mean_val = values.mean()
            std_val = values.std()
            threshold = mean_val + γ * std_val
            mask = values >= threshold
            new_values = values * mask
            adj_list_new.append(torch.sparse_coo_tensor(adj._indices(), new_values, adj.shape, device=device).coalesce())
        adj_H = build_affinity_matrix(H, args.neighbor_num, device).to(device)
        result_list = []
        beta = 0.8   # 0.2 0.4 0.6 0.8 1.0
        result_list = [
        beta *(adj_H.to_dense() * adj_list[i].to_dense()) + (1 - beta) * adj_list_new[i].to_dense()
        for i in range(len(adj_list))
        ]
        result_list = [result.to_dense() for result in result_list]
        
        with torch.no_grad():
            fused_graph,weights = graph_fusion(result_list)
        loss_list = []
        for v in range(view):
            loss_list.append(weights[v]* contrastiveloss(H, Hs[v], fused_graph))  
            loss_list.append(mse(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        torch.cuda.empty_cache()
        
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
    return loss


accs = []
nmis = []
purs = []
if not os.path.exists('./models'):
    os.makedirs('./models')
T = 1
for i in range(T):
    print("ROUND:{} DataName:{} view_num:{}".format(i + 1, Dataname, view))
    setup_seed(seed)
    model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num,device)
    model = model.to(device)
    device = next(model.parameters()).device 
    state = model.state_dict()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    contrastiveloss = ContrastiveLoss(args.batch_size, args.temperature, args.neighbor_num, class_num, device).to(device)
    best_acc, best_nmi, best_pur = 0, 0, 0
    epoch = 1
    neighbor_num = args.neighbor_num
    while epoch <= args.pre_epochs:
        pretrain(epoch)
        epoch += 1
    #acc, nmi, ari, pur = valid(model, device, dataset, view, data_size, class_num, args=args, eval_h=True, epoch=epoch)

    while epoch <= args.pre_epochs + args.con_epochs:
        loss = contrastive_train(epoch)
        acc, nmi, ari, pur = valid(model, device, dataset, view, data_size, class_num, args=args, eval_h=False, epoch=epoch)     
  
        if acc > best_acc:
            best_acc, best_nmi, best_pur = acc, nmi, pur
            best_epoch = epoch
            state = model.state_dict()
            torch.save(state, f'./models/{args.dataset}.pth')
        epoch += 1
        
    accs.append(best_acc)
    nmis.append(best_nmi)
    purs.append(best_pur)
    print('The best k-means clustering performace at epoch {} : ACC = {:.4f} NMI = {:.4f} PUR={:.4f}'.format(best_epoch, best_acc, best_nmi, best_pur))


