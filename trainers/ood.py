import numpy as np
import time
from metrics import averageMeter
from trainers.base import BaseTrainer
from trainers.logger import BaseLogger
from optimizers import get_optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch
from torch.optim import Adam
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
from utils import roc_btw_arr, batch_run
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
from schedulers import WarmUpLR
from loader import acoustic_dataset


class OODTrainer(BaseTrainer):
    def train(self, model, opt, d_dataloaders, logger=None, logdir='', scheduler=None, clip_grad=None):
        cfg = self.cfg
        self.logdir = logdir
        best_val_loss = np.inf
        time_meter = averageMeter()
        i = 0
        indist_train_loader = d_dataloaders['indist_train']
        indist_val_loader = d_dataloaders['indist_val']
        
        for i_epoch in range(cfg.n_epoch):

            for x, y in indist_train_loader:
                i += 1

                model.train()
                x = x.to(self.device)
                y = y.to(self.device)

                start_ts = time.time()
                d_train = model.train_step(x, y=y, optimizer=opt, clip_grad=clip_grad)
                time_meter.update(time.time() - start_ts)
                logger.process_iter_train(d_train)

                if isinstance(scheduler, _LRScheduler):
                    if not (isinstance(scheduler, ReduceLROnPlateau) or isinstance(scheduler, WarmUpLR)):
                        scheduler.step()

                if i % cfg.print_interval == 0:
                    d_train = logger.summary_train(i)
                    print(f"Iter [{i:d}] Avg Loss: {d_train['loss/train_loss_']:.4f} Elapsed time: {time_meter.sum:.4f}")
                    time_meter.reset()

                if i % cfg.val_interval == 0:
                    model.eval()
                    for val_x, val_y in indist_val_loader:
                        val_x = val_x.to(self.device)
                        val_y = val_y.to(self.device)

                        d_val = model.validation_step(val_x, y=val_y)
                        logger.process_iter_val(d_val)
                        if cfg.get('val_once', False):
                            # no need to run the whole val set
                            break

                    '''AUC'''
                    d_result = self.get_auc(model, d_dataloaders)
                    logger.d_val.update(d_result)

                    d_val = logger.summary_val(i)
                    val_loss = d_val['loss/val_loss_']
                    print(d_val['print_str'])
                    best_model = val_loss < best_val_loss

                    if cfg.save_interval is not None and i % cfg.save_interval == 0 and best_model:
                        self.save_model(model, logdir, best=best_model, i_iter=i, i_epoch=i_epoch)
                        print(f'Iter [{i:d}] best model saved {val_loss} <= {best_val_loss}')
                        best_val_loss = val_loss
                    if isinstance(scheduler, _LRScheduler):
                        if isinstance(scheduler, ReduceLROnPlateau):
                            scheduler.step(val_loss)
                        elif isinstance(scheduler, WarmUpLR):
                            scheduler.step()
                        else:
                            scheduler.step()

            if 'save_interval_epoch' in cfg and i_epoch % cfg.save_interval_epoch == 0:
                self.save_model(model, logdir, best=False, i_epoch=i_epoch)

        '''AUC'''
        model.eval()

        d_result = self.get_auc(model, d_dataloaders)
        print(d_result)

        return model, d_result 

    def predict(self, m, dl, device, flatten=False):
        """run prediction for the whole dataset"""
        l_result = []
        for x, _ in dl:
            if flatten:
               x = x.view(len(x), -1)
            pred = m.predict(x.cuda(device)).detach().cpu()
            l_result.append(pred)
        return torch.cat(l_result)

    def get_auc(self, model, d_dataloaders):
        indist_val_loader = d_dataloaders['indist_val']
        in_pred = self.predict(model, indist_val_loader, self.device)
        dcase = False
        d_result = {}
        for k, v in d_dataloaders.items():
            if k.startswith('ood_'):
                ood_pred = batch_run(model, v, self.device)
                auc = roc_btw_arr(ood_pred, in_pred)
                k = k.replace('ood_', '')
                d_result[f'result/auc_{k}_'] = auc

            if k.startswith('dcase_'):
                auc, pauc = acoustic_dataset.compute_auc_per_wav_per_id(model, v, self.device) 
                d_result[f'result/auc_{k}_'] = auc
                d_result[f'result/pauc_{k}_'] = pauc
                dcase = True
        # add overall average for dcase
        if dcase:
            auc = np.mean([v for k, v in d_result.items() if k.startswith('result/auc_dcase_')])
            d_result['result/auc_dcase_'] = auc
            pauc = np.mean([v for k, v in d_result.items() if k.startswith('result/pauc_dcase_')])
            d_result['result/pauc_dcase_'] = pauc
        torch.save(d_result, self.logdir + '/val_auc.pkl')

        return d_result 

