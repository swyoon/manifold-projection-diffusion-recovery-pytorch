from trainers.logger import BaseLogger, ClsLogger
from trainers.base import BaseTrainer
from trainers.ood import OODTrainer
from trainers.mpdr import MPDRTrainer


def get_trainer(cfg):
    # get trainer by specified `trainer` field
    # if not speficied, get trainer by model type
    trainer_type = cfg.get('trainer', None)
    arch = cfg['model']['arch']
    device = cfg['device']
    if trainer_type == 'ood':
        trainer = OODTrainer(cfg['training'], device=device)
    elif trainer_type == 'nae':
        trainer = NAETrainer(cfg['training'], device=device)
    elif trainer_type == 'nae_cl':
        trainer = NAECLTrainer(cfg['training'], device=device)
    elif trainer_type == 'mpdr':
        trainer = MPDRTrainer(cfg['training'], device=device)
    else:
        trainer = BaseTrainer(cfg['training'], device=device)
    return trainer


def get_logger(cfg, writer):
    logger_type = cfg['logger']
    logger = BaseLogger(writer)
    return logger 
