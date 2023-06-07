import os
import numpy as np
import torch
from torch.optim import Adam
import time
from trainers.base import BaseTrainer
from metrics import averageMeter
from tqdm import tqdm, trange
from utils import roc_btw_arr, batch_run
from models.on_manifold_drl import (
    MPDR, MPDR_Ensemble, 
)
from models.on_manifold_drl.v5_single import MPDR_Single
from loader import acoustic_dataset


class DivergenceDetector():
    """
    detects divergence in the training process from observing negative sample energy.
    Collect negative sample energy and compute Gaussian statistics by rolling mean and rolling standard deviation.
    Declare divergence when negative sample energy is larger than 6-sigma interval from the mean.
    """
    def __init__(self, window_size=1000, disable=False):
        self.window_size = window_size
        self.neg_sample_energy = []
        self.mean = 0
        self.std = 0
        self.disable = disable

    def update(self, neg_sample_energy):
        self.neg_sample_energy.append(neg_sample_energy)
        if len(self.neg_sample_energy) > self.window_size:
            self.neg_sample_energy.pop(0)
        self.mean = np.mean(self.neg_sample_energy)
        self.std = np.std(self.neg_sample_energy)

    def detect(self):
        if self.disable:
            return False
        if len(self.neg_sample_energy) < self.window_size:
            return False
        if self.neg_sample_energy[-1] > self.mean + 6 * max(self.std, 0.1) and self.neg_sample_energy[-1] > 0.2:
            return True
        return False


class LatentMPDRTrainer(BaseTrainer):
    def train(self, model, opt, d_dataloaders, logger=None, logdir='', scheduler=None, clip_grad=None):
        cfg = self.cfg
        best_val_loss = np.inf
        time_meter = averageMeter()
        i = 0
        indist_train_loader = d_dataloaders['indist_train']
        indist_val_loader = d_dataloaders['indist_val']
        # oodval_val_loader = d_dataloaders['ood_val']
        # oodtarget_val_loader = d_dataloaders['ood_target']
        divergence = DivergenceDetector(disable=cfg.get('disable_divergence_detector', False))
        self.logdir = logdir

        n_ae_epoch = cfg.ae_epoch
        n_nae_epoch = cfg.nae_epoch
        n_netx_epoch = cfg.get('netx_epoch', 0)
        ae_opt, mpdr_opt, netx_opt = self.get_opt(model)

        ########################################
        '''AE PASS'''
        ########################################
        if cfg.get('load_ae', None) is not None:
            n_ae_epoch = 0
            self.load_ae(model)

        logger.has_train_loss = False  # autoencoder train loss is recorded separately 
        for i_epoch in trange(n_ae_epoch, bar_format='{l_bar}{bar:20}{r_bar}'):

            for x, y in indist_train_loader:
                i += 1

                model.train()
                x = x.to(self.device)
                y = y.to(self.device)

                start_ts = time.time()
                d_train = model.train_step_ae(x, ae_opt, clip_grad=clip_grad)
                time_meter.update(time.time() - start_ts)
                logger.process_iter_train(d_train)

                if i % cfg.print_interval == 0:
                    d_train = logger.summary_train(i)
                    print(f"Iter [{i:d}] Avg Loss: {d_train['loss/train_loss_ae_']:.4f} Elapsed time: {time_meter.sum:.4f}")
                    time_meter.reset()

                if i % cfg.val_interval == 0:
                    model.eval()
                    for val_x, val_y in indist_val_loader:
                        val_x = val_x.to(self.device)
                        val_y = val_y.to(self.device)

                        d_val = model.validation_step_ae(val_x, y=val_y)
                        logger.process_iter_val(d_val)
                    d_val = logger.summary_val(i)
                    val_loss = d_val['loss/val_loss_']
                    print(d_val['print_str'])
                    best_model = val_loss < best_val_loss

                    if best_model:
                        self.save_model(model, logdir, best=best_model, i_iter=i)
                        print(f'Iter [{i:d}] best model saved {val_loss} <= {best_val_loss}')
                        best_val_loss = val_loss

            if i_epoch % cfg.get('save_interval_epoch', 1) == 1:
                self.save_model(model, logdir, best=False, i_iter=None, i_epoch=i_epoch)
                print(f'Epoch [{i_epoch:d}] model saved for epoch {i_epoch}')

        if n_ae_epoch > 0:
            # save final ae model
            self.save_model(model, logdir, best=False, i_iter=None, i_epoch=None, last=True)
            print(f'Epoch [{i_epoch:d}] model saved ')
        if cfg.get('load_best_ae', False):
            model.load_state_dict(torch.load(os.path.join(logdir, 'model_best.pkl'))['model_state'])
            

        ##################################################################
        ''' NET X PASS'''
        ##################################################################

        if cfg.get('init_net_x_pretrained', None) is not None:
            # load net_x from separately pre-trained checkpoint (e.g. training IDNN separately)
            state_dict = torch.load(cfg['init_net_x_pretrained'], map_location=self.device)['model_state']
            model.net_x.load_state_dict(state_dict)
            n_netx_epoch = 0
            print(f'net x loaded {cfg["init_net_x_pretrained"]}')

        if cfg.get('init_net_x', None) is not None:
            # load net_x from previous MPDR run
            state_dict = torch.load(cfg['init_net_x'], map_location=self.device)['model_state']
            state_dict = {k.replace('net_x.', ''): v for k, v in state_dict.items() if 'net_x.' in k}
            model.net_x.load_state_dict(state_dict)
            n_netx_epoch = 0
            print(f'net x loaded {cfg["init_net_x"]}')


        i = 0
        for i_epoch in tqdm(range(n_netx_epoch), bar_format='{l_bar}{bar:20}{r_bar}'):
            for x, _ in tqdm(indist_train_loader, bar_format='{l_bar}{bar:20}{r_bar}'):
                i += 1

                x = x.to(self.device)
                d_result = model.net_x.train_step(x, netx_opt)
                logger.process_iter_train(d_result)

                if i % cfg.print_interval_netx == 1:
                    logger.summary_train(i)

        if n_netx_epoch > 0:
            # save final ae+netx model
            self.save_model(model, logdir, best=False, i_iter=None, i_epoch=None, last=True)
            print(f'Epoch [{i_epoch:d}] model saved ')


        ###################################################################
        '''MPDR PASS'''
        ##################################################################
        logger.has_train_loss = True 
        logger.has_val_loss = False
        if cfg.get('transfer_ae', False):
            # initialize net_x autoencoder from model.ae
            if hasattr(model.net_x, 'ae'):
                model.net_x.ae.load_state_dict(model.ae.state_dict())
                print('net_x.ae initialized from ae of mpdr')
            else:
                model.net_x.encoder.load_state_dict(model.ae.encoder.state_dict())
                model.net_x.decoder.load_state_dict(model.ae.decoder.state_dict())
                print('net_x initialized from ae of mpdr')

        i = 0
        lr = cfg.nae_lr

        # resume training from the last checkpoint
        if cfg.get('load_model', None) is not None:
            model, mpdr_opt, i, i_epoch = self.load_model(model, mpdr_opt, cfg['load_model'])

        logger.writer.add_scalar('learning_rate', lr, i)
        for i_epoch in tqdm(range(n_nae_epoch), bar_format='{l_bar}{bar:20}{r_bar}'):
            for x, _ in tqdm(indist_train_loader, bar_format='{l_bar}{bar:20}{r_bar}'):
                i += 1

                # learning rate warm-up
                if i < cfg.get('warmup_iter', 0):
                    lr = cfg.nae_lr * min(1., i / cfg.warmup_iter)
                    for param_group in mpdr_opt.param_groups:
                        param_group['lr'] = lr
                    if i % 100 == 1:
                        logger.writer.add_scalar('learning_rate', lr, i)

                # learning rate decay
                if i in cfg.get('lr_decay_iter', []):
                    lr = lr * 0.2
                    for param_group in mpdr_opt.param_groups:
                        param_group['lr'] = lr
                    logger.writer.add_scalar('learning_rate', lr, i)

                x = x.cuda(self.device)
                if cfg.get('flatten', False):
                    x = x.view(len(x), -1)
                d_result = model.train_step(x, mpdr_opt, clip_grad=clip_grad, mode=cfg.get('mode', 'off'))
                logger.process_iter_train(d_result)

                neg_e = d_result[f'mpdr/{cfg.mode}_neg_e_']
                divergence.update(neg_e)
                if i > 2000 and divergence.detect():
                    print(f'divergence detected: {neg_e} >> {divergence.mean} + 6 * {divergence.std}')
                    break 

                if i % cfg.print_interval_nae == 1:
                    logger.summary_train(i)
 
                if i % cfg.val_interval_nae == 1:
                    '''AUC'''
                    d_result = self.get_auc(model, d_dataloaders)

                    if (cfg.get('sample_val_interval', None) is not None) \
                            and i % (cfg.val_interval_nae * cfg.sample_val_interval) == 1 :
                        """generate samples. do this less frequently than typical validation"""
                        if hasattr(model, 'sample_step'):
                            d_sample = model.sample_step(x)
                            d_result = {**d_result, **d_sample}

                    logger.process_iter_val(d_result)
                    print(logger.summary_val(i)['print_str'])

            if divergence.detect():
                break 

            self.save_model_ompd(model, mpdr_opt, logdir, i_iter=i, i_epoch=i_epoch)
        if divergence.detect():
            # retrieve the best model
            if cfg.get('debug', False):
                from pudb import set_trace; set_trace()
            model, _, _, _ = self.load_model(model, mpdr_opt, logdir + '/mpdr.pkl')
            print('best model loaded')
        else:  # if not diverged, save the last model
            self.save_model_ompd(model, mpdr_opt, logdir, i_iter=i, i_epoch=i_epoch)

        '''AUC'''
        d_result = self.get_auc(model, d_dataloaders)
        print(d_result)
        # save validation result to logdir


        return model, d_result

    def load_ae(self, model):
        cfg = self.cfg
        state_dict = torch.load(cfg['load_ae'])['model_state']
        if isinstance(model, MPDR):
            encoder_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder')}
            decoder_state = {k.replace('decoder.', ''): v for k, v in state_dict.items() if k.startswith('decoder')}
            model.encoder.load_state_dict(encoder_state)
            model.decoder.load_state_dict(decoder_state)
        elif isinstance(model, MPDR_Ensemble):
            state_dict = torch.load(cfg['load_ae'])['model_state']
            ae_states = {k:v for k, v in state_dict.items() if k.startswith('l_ae')}
            model.load_state_dict(ae_states, strict=False)
        elif isinstance(model, MPDR_Joint) or isinstance(model, MPDR_Single):
            state_dict = torch.load(cfg['load_ae'])['model_state']
            ae_states = {k:v for k, v in state_dict.items() if k.startswith('ae')}
            model.load_state_dict(ae_states, strict=False)
        elif isinstance(model, MPDR_Conditional_IDNN):
            state_dict = torch.load(cfg['load_ae'])['model_state']
            ae_states = {k:v for k, v in state_dict.items() if k.startswith('ae')}
            model.load_state_dict(ae_states, strict=False)
        else:
            raise NotImplementedError
        print(f'model loaded from {cfg["load_ae"]}')
    
    def get_opt(self, model):
        cfg = self.cfg
        netx_opt = None
        if isinstance(model, MPDR):
            ae_opt = Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=cfg.ae_lr)
            mpdr_opt = Adam(list(model.net_x.parameters()) + list(model.net_z.parameters()), lr=cfg.nae_lr)
        elif isinstance(model, MPDR_Ensemble):
            ae_opt = [Adam(list(ae.parameters()), lr=cfg.ae_lr) for ae in model.l_ae]
            mpdr_opt = Adam(list(model.net_x.parameters()) + list(model.l_net_z.parameters()), lr=cfg.nae_lr)
        elif isinstance(model, MPDR_Joint):
            ae_opt = Adam(list(model.ae.parameters()), lr=cfg.ae_lr)
            params = []
            if cfg.get('mode', 'off'):
                if model.net_x is not None:
                    params += list(model.net_x.parameters())
                if model.net_z is not None:
                    params += list(model.net_z.parameters())
                if model.use_recon_error:
                    params += [model.a, model.b]
            elif cfg.mode == 'on':
                params += list(model.net_s.parameters())
            else:
                raise NotImplementedError

            mpdr_opt = Adam(params, lr=cfg.nae_lr)
        elif isinstance(model, MPDR_Single):
            ae_opt = Adam(list(model.ae.parameters()), lr=cfg.ae_lr)
            if 'netx_epoch' in cfg:
                netx_opt = Adam(list(model.net_x.parameters()), lr=cfg.netx_lr)

            if cfg.get('opt', None) == 'fix_D':  # fix decoder
                mpdr_opt = Adam(list(model.net_x.encoder.parameters()) + list(model.net_x.net_b.parameters()), lr=cfg.nae_lr)
            else:
                mpdr_opt = Adam(list(model.net_x.parameters()), lr=cfg.nae_lr)
        else:
            raise NotImplementedError
        return ae_opt, mpdr_opt, netx_opt

    def predict(self, m, dl, device, flatten=False):
        """run prediction for the whole dataset"""
        l_result = []
        for x, _ in dl:
            with torch.no_grad():
                if flatten:
                    x = x.view(len(x), -1)
                pred = m.predict(x.cuda(device)).detach().cpu()
            l_result.append(pred)
        return torch.cat(l_result)

    def save_model_ompd(self, model, opt, logdir, i_iter, i_epoch):
        state = {'model_state': model.state_dict(), 'opt_state': opt.state_dict(),
                 'i': i_iter, 'i_epoch': i_epoch}
        fname = f'mpdr_{i_epoch}.pkl'
        torch.save(state, os.path.join(logdir, fname))
        print(f'model saved at {fname}')

    def load_model(self, model, opt, path):
        state = torch.load(path)
        model.load_state_dict(state['model_state'])
        opt.load_state_dict(state['opt_state'])
        print(f'model loaded from {path}')
        return model, opt, state['i'], state['i_epoch']

    def get_auc(self, model, d_dataloaders):
        indist_val_loader = d_dataloaders['indist_val']
        in_pred = self.predict(model, indist_val_loader, self.device)
        d_result = {}
        dcase = False
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

