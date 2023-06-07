
import os
import random
import argparse
from omegaconf import OmegaConf
import numpy as np
from itertools import cycle
import torch
from models import get_model
from trainers import get_trainer, get_logger
from loader import get_dataloader
from optimizers import get_optimizer
from schedulers import get_scheduler
from datetime import datetime
from tensorboardX import SummaryWriter
from utils import save_yaml, search_params_intp, eprint, parse_unknown_args, parse_nested_args
from gpu_utils import AutoGPUAllocation
import wandb


def run(cfg, writer, use_nni=False):
    """main training function"""
    # Setup seeds
    seed = cfg.get('seed', 1)
    print(f'running with random seed : {seed}')
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # for reproducibility
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # Setup device
    device = cfg.device

    # Setup Dataloader
    d_dataloaders = {}
    for key, dataloader_cfg in cfg['data'].items():
        if 'holdout' in cfg:
            dataloader_cfg = process_holdout(dataloader_cfg, int(cfg['holdout']))
        d_dataloaders[key] = get_dataloader(dataloader_cfg)

    # Setup Model
    model = get_model(cfg).to(device)
    trainer = get_trainer(cfg)
    logger = get_logger(cfg, writer)

    # Setup optimizer
    if hasattr(model, 'own_optimizer') and model.own_optimizer:
        optimizer, sch = model.get_optimizer(cfg['training']['optimizer'])
    elif 'optimizer' not in cfg['training']:
        optimizer = None
        sch = None
    else:
        optimizer, sch = get_optimizer(cfg["training"]["optimizer"], model.parameters())

    # lr scheduler 
    # sch = get_scheduler(optimizer, cfg['training']['lr_schedule'])

    model, train_result = trainer.train(model, optimizer, d_dataloaders, logger=logger,
                                   logdir=writer.file_writer.get_logdir(), scheduler=sch,
                                   clip_grad=cfg['training'].get('clip_grad', None))
    if use_nni:
        nni.report_final_result(train_result)


def make_run_id(nni_params):
    l_run_id = []
    for key, val in nni_params.items():
        l_run_id.append(str(key) + str(val))
    run_id = '-'.join(sorted(l_run_id))
    return run_id


def process_holdout(dataloader_cfg, holdout):
    """udpate config if holdout option is present in config"""
    if 'LeaveOut' in dataloader_cfg['dataset'] and 'out_class' in dataloader_cfg:
        dataloader_cfg['out_class'] = holdout
    print(dataloader_cfg)
    return dataloader_cfg



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--nni', action='store_true')
    parser.add_argument('--device', default=0)
    parser.add_argument('--logdir', default='results/')
    parser.add_argument('--run', default=None, help='unique run id of the experiment')
    parser.add_argument('--config2', type=str, default=None, help='additional config file')
    parser.add_argument('--config3', type=str, default=None, help='additional config file')
    args, unknown = parser.parse_known_args()
    d_cmd_cfg = parse_unknown_args(unknown)
    d_cmd_cfg = parse_nested_args(d_cmd_cfg)
    print(d_cmd_cfg)
    cfg = OmegaConf.load(args.config)

    # load additional config file and merge with main config
    if args.config2 is not None:
        cfg2 = OmegaConf.load(args.config2)
        cfg = OmegaConf.merge(cfg, cfg2)
    if args.config3 is not None:
        cfg3 = OmegaConf.load(args.config3)
        cfg = OmegaConf.merge(cfg, cfg3)

    if args.device == 'cpu':
        cfg['device'] = f'cpu'
    elif args.device == 'auto':
        gpu_allocation = AutoGPUAllocation()
        device = gpu_allocation.device
        cfg['device'] = device
    else:
        cfg['device'] = f'cuda:{args.device}'

    if args.nni:
        import nni
        nni_params = nni.get_next_parameter()
        eprint(nni_params)
        # make run id
        run_id = make_run_id(nni_params)

        nni_params = search_params_intp(nni_params)
        eprint(nni_params)
        cfg = OmegaConf.merge(cfg, nni_params)
        eprint(OmegaConf.to_yaml(cfg))
    else:
        if args.run is None:
            run_id = datetime.now().strftime('%Y%m%d-%H%M')
        else:
            run_id = args.run
    cfg = OmegaConf.merge(cfg, d_cmd_cfg)
    print(OmegaConf.to_yaml(cfg))

    config_basename = os.path.basename(args.config).split('.')[0]
    logdir = os.path.join(args.logdir, config_basename, str(run_id))
    writer = SummaryWriter(logdir=logdir)
    print("Result directory: {}".format(logdir))
    # wandb.init(project=args.logdir.split('/')[-1], config=cfg, job_type='train',
    #            name='_'.join([config_basename, str(run_id)]), dir=logdir)

    # copy config file
    copied_yml = os.path.join(logdir, os.path.basename(args.config))
    save_yaml(copied_yml, OmegaConf.to_yaml(cfg))
    print(f'config saved as {copied_yml}')

    run(cfg, writer, use_nni=args.nni)


