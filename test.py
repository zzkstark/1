import torch
from network import Network
from metric import valid
import argparse
from dataloader import load_data

Dataname = 'Caltech-5V'
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset, dims, view, data_size, class_num = load_data(args.dataset)
model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num,device)
model = model.to(device)

checkpoint = torch.load('./models/' + args.dataset + '.pth')
model.load_state_dict(checkpoint)

model.eval()
print("Dataset:{}".format(args.dataset))
print("Datasize:" + str(data_size))
print("Loading models...")
acc, nmi, ari, pur  = valid(model, device, dataset, view, data_size, class_num, args, eval_h=False)

