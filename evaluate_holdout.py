"""evaluation script for class hold-out experiment"""
import os
import yaml
import argparse
import copy
import torch
from models import get_model
from loader import get_dataloader
from gpu_utils import AutoGPUAllocation

from utils import roc_btw_arr, batch_run, aupr_btw_arr 

parser = argparse.ArgumentParser()
parser.add_argument('--resultdir', type=str, help='result dir')
parser.add_argument('--config', type=str, help='config file name')
parser.add_argument('--ood', type=str, help='list of OOD datasets, separated by comma')
parser.add_argument('--ckpt', type=str, help='checkpoint file name to load. default', default=None)
parser.add_argument('--device', type=str, help='device', default=0)
parser.add_argument('--dataset', type=str, choices=['MNIST', 'CIFAR10'],
                    default='MNIST', help='dataset to use : MNIST, CIFAR10')
parser.add_argument('--tag_epoch', action='store_true', help='tag epoch in result file name')
args = parser.parse_args()


# cfg = yaml.load(open(args.config), Loader=yaml.FullLoader)
cfg = yaml.load(open(os.path.join(args.resultdir, args.config)), Loader=yaml.FullLoader)
model_type = args.config.split('.')[0]  # ex. mnist_ho_ae
if model_type == 'mnist_ho_vael':
    model_name = 'vae'
else:
    # model_name = model_type.split('_')[-1]  # ex. ae
    model_name = model_type
# param = os.path.basename(args.config).split('.')[0]
# results/nni_mnist_ho_ebae/ebae/holdout0model.z_dim16
# result_dir = f'results/{args.dataset.lower()}_holdout/vs{args.holdout}/{model_type}/{param}/{args.run}/'
result_dir = args.resultdir
if args.ckpt is not None:
    ckpt_file = os.path.join(result_dir, args.ckpt)
elif 'ebae' in cfg['model']['arch']:
    ckpt_file = os.path.join(result_dir, f'ebae.pkl')
else:
    raise ValueError(f'ckpt file not specified')
print(f'loading from {ckpt_file}')
if args.ood is not None:
    l_ood = [s.strip() for s in args.ood.split(',')]
else:
    l_ood = []

if args.device == 'cpu':
    device = f'cpu'
elif args.device == 'auto':
    gpu_allocation = AutoGPUAllocation()
    device = gpu_allocation.device
else:
    device = f'cuda:{args.device}'


print(f'loading from : {ckpt_file}')

if args.tag_epoch:
    epoch = int(args.ckpt.split('.')[0].split('_')[-1])


# load dataset
# cfg['holdout'] = 9
# holdout = int(args.holdout)
holdout = int(cfg['holdout'])
print(f'holdout {holdout}')
indist_dict = {'dataset': f'{args.dataset}LeaveOut',
               'batch_size': 128,
               'n_workers': 8,
               'path': 'datasets',
               'shuffle': False,
               'split': 'evaluation',
               'out_class': holdout,
               'holdout': False}


ood_dict = {'dataset': f'{args.dataset}LeaveOut',
            'batch_size': 128,
            'n_workers': 8,
            'path': 'datasets',
            'shuffle': False,
            'split': 'evaluation',
            'out_class': holdout,
            'holdout': True}

in_dl = get_dataloader(indist_dict)
ood_dl = get_dataloader(ood_dict)


print('ood datasets')
print(l_ood)
if args.dataset == 'MNIST':
    channel = 1
    size = 28
else:
    channel = 3
    size = 32
data_dict = {'dataset': None,
             'batch_size': 128,
             'n_workers': 8,
             'size': size,
             'channel': channel,
             'path': 'datasets',
             'shuffle': False,
             'split': 'evaluation'}

l_ood_dl = []
for ood_name in l_ood:
    data_dict_ = copy.copy(data_dict)
    data_dict_['dataset'] = ood_name
    dl = get_dataloader(data_dict_)
    l_ood_dl.append(dl)



# load model
model = get_model(cfg).to(device)
ckpt_data = torch.load(ckpt_file)
if 'model_state' in ckpt_data:
    model.load_state_dict(ckpt_data['model_state'])
else:
    model.load_state_dict(ckpt_data)

# generate file containing AUC performance
in_pred = batch_run(model, in_dl, device=device, no_grad=False)
ood_pred = batch_run(model, ood_dl, device=device, no_grad=False)

l_ood_pred = []
for dl in l_ood_dl:
    l_ood_pred.append(batch_run(model, dl, device=device, no_grad=False))

ood_auc = roc_btw_arr(ood_pred, in_pred)
ood_aupr = aupr_btw_arr(ood_pred, in_pred)
l_ood_auc = []
l_ood_aupr = []
for pred in l_ood_pred:
    l_ood_auc.append(roc_btw_arr(pred, in_pred))
    l_ood_aupr.append(aupr_btw_arr(pred, in_pred))

print('inlier recon error', in_pred.mean())
print('AUROC', ood_auc)
print('AUPRC', ood_aupr)
print(l_ood_auc)

if args.tag_epoch:
    result_auroc = f'{args.dataset}_holdout{holdout}_auroc_epoch{epoch}.txt'
    result_aupr = f'{args.dataset}_holdout{holdout}_aupr_epoch{epoch}.txt'
else:
    result_auroc = f'{args.dataset}_holdout{holdout}_auroc.txt'
    result_aupr = f'{args.dataset}_holdout{holdout}_aupr.txt'

with open(os.path.join(result_dir, result_auroc), 'w') as f:
    f.write(str(ood_auc))

with open(os.path.join(result_dir, result_aupr), 'w') as f:
    f.write(str(ood_aupr))

for ood_name, auc, aupr in zip(l_ood, l_ood_auc, l_ood_aupr):
    with open(os.path.join(result_dir, f'{ood_name}.txt'), 'w') as f:
        f.write(str(auc))
    with open(os.path.join(result_dir, f'{ood_name}_aupr.txt'), 'w') as f:
        f.write(str(aupr))
    with open(os.path.join(result_dir, f'{ood_name}_aupr_epoch{epoch}.txt'), 'w') as f:
        f.write(str(aupr))
