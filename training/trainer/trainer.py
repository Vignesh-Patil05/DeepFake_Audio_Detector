import os
import pickle
import datetime
import logging
import numpy as np
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from metrics.base_metrics_class import Recorder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    from .sam import SAM
    from .bypass_bn import disable_running_stats, enable_running_stats
except:
    from sam import SAM
    from bypass_bn import disable_running_stats, enable_running_stats

class Trainer(object):
    def __init__(
        self, 
        config, 
        model, 
        optimizer, 
        scheduler,
        logger,
        metric_scoring='auc',
        ):
        # check if all the necessary components are implemented
        if config is None or model is None or optimizer is None or logger is None:
            raise ValueError("config, model, optimizier, logger, and tensorboard writer must be implemented")
        
        self.config = config
        self.model = model
        
        opt_name = config['optimizer']['type']
        if opt_name == 'sgd':
            base_optimizer = torch.optim.SGD
            self.optimizer = SAM(self.model.parameters(), base_optimizer, 
                                rho=config['sam_param']['rho'], adaptive=config['sam_param']['use_adaptive_sam'], 
                                lr=float(config['optimizer'][opt_name]['lr']), 
                                momentum=config['optimizer'][opt_name]['momentum'], 
                                weight_decay=config['optimizer'][opt_name]['weight_decay'])
        elif opt_name == 'adam':
            base_optimizer = torch.optim.Adam
            self.optimizer = SAM(self.model.parameters(), base_optimizer, 
                                rho=config['sam_param']['rho'], adaptive=config['sam_param']['use_adaptive_sam'], 
                                lr=float(config['optimizer'][opt_name]['lr']), 
                                weight_decay=config['optimizer'][opt_name]['weight_decay'],
                                betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
                                eps=config['optimizer'][opt_name]['eps'],
                                amsgrad=config['optimizer'][opt_name]['amsgrad'],
            )

        else:
            raise NotImplementedError('Optimizer {} is not implemented'.format(config['optimizer']))
    
        if config['lr_scheduler'] is None:
            self.scheduler = None
        elif config['lr_scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=config['lr_step'], 
                gamma=config['lr_gamma'],
            )
        elif config['lr_scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=config['lr_T_max'], 
                eta_min=config['lr_eta_min'],
            )
        else:
            raise NotImplementedError('Scheduler {} is not implemented'.format(config['lr_scheduler']))
        
        
        self.writers = {}  # dict to maintain different tensorboard writers for each dataset and metric
        self.logger = logger
        self.metric_scoring = metric_scoring
        # maintain the best metric of all epochs
        self.best_metrics_all_time = defaultdict(
            lambda: defaultdict(lambda: float('-inf') 
            if self.metric_scoring != 'eer' else float('inf'))
        ) 
        self.speed_up()  # move model to GPU

        # get current time
        self.timenow = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        # create directory path
        self.log_dir = os.path.join(
            self.config['log_dir'], 
            self.config['model_name'] + '_' + self.timenow
        )
        os.makedirs(self.log_dir, exist_ok=True)
        self.marker = 0

    def get_writer(self, phase, dataset_key, metric_key):
        writer_key = f"{phase}-{dataset_key}-{metric_key}"
        if writer_key not in self.writers:
            # update directory path
            writer_path = os.path.join(
                self.log_dir,
                phase, 
                dataset_key,
                metric_key
            )
            os.makedirs(writer_path, exist_ok=True)
            # update writers dictionary
            self.writers[writer_key] = SummaryWriter(writer_path)
        return self.writers[writer_key]

    def speed_up(self):
        if self.config['ngpu'] > 1:
            self.model = DataParallel(self.model)
        self.model.to(device)
    
    def setTrain(self):
        self.model.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def load_ckpt(self, model_path):
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            suffix = model_path.split('.')[-1]
            if suffix == 'p':
                self.model.load_state_dict(saved.state_dict())
            else:
                self.model.load_state_dict(saved)
            self.logger.info('Model found in {}'.format(model_path))
        else:
            raise NotImplementedError(
                "=> no model found at '{}'".format(model_path))

    def save_ckpt(self, phase, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"ckpt_best.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        if self.config['ngpu'] > 1:
            torch.save(self.model.module.state_dict(), save_path)
        else:
            torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Checkpoint saved to {save_path}")

    def save_feat(self, phase, pred_dict, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        features = pred_dict['feat']
        feat_name = f"feat_best.npy"
        save_path = os.path.join(save_dir, feat_name)
        np.save(save_path, features.cpu().numpy())
        self.logger.info(f"Feature saved to {save_path}")
    
    def save_data_dict(self, phase, data_dict, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'data_dict_{phase}.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(data_dict, file)
        self.logger.info(f"data_dict saved to {file_path}")

    def save_metrics(self, phase, metric_one_dataset, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, 'metric_dict_best.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(metric_one_dataset, file)
        self.logger.info(f"Metrics saved to {file_path}")

    def train_epoch(
        self, 
        epoch, 
        train_data_loader, 
        test_data_loaders=None,
        ):

        self.logger.info("===> Epoch[{}] start!".format(epoch))
        test_step = 1
        step_cnt = epoch * len(train_data_loader)

        # save the training data_dict
        data_dict = train_data_loader.dataset.data_dict
        self.save_data_dict('train', data_dict, ','.join(self.config['train_dataset']))
        
        # define training recorder
        train_recorder_loss = defaultdict(Recorder)
        train_recorder_metric = defaultdict(Recorder)

        for iteration, data_dict in tqdm(enumerate(train_data_loader)):
            self.setTrain()

            # get data
            data, label = \
                data_dict['audio'], data_dict['label']
            if 'label_spe' in data_dict:
                label_spe = data_dict['label_spe']
                data_dict['label_spe'] = label_spe.to(device)
            
            # move data to GPU
            data_dict['audio'], data_dict['label'] = data.to(device), label.to(device)

            # SAM Implementation
            enable_running_stats(self.model)
            predictions = self.model(data_dict)
            losses = self.model.get_losses(data_dict, predictions)
            losses['overall'].backward()
            
            self.optimizer.first_step(zero_grad=True)

            # second forward-backward step
            disable_running_stats(self.model)
            predictions = self.model(data_dict) # model-prime
            losses = self.model.get_losses(data_dict, predictions)
            losses['overall'].backward()

            self.optimizer.second_step(zero_grad=True)
            
            if self.scheduler is not None:
                self.scheduler.step()
                
            # compute training metric for each batch data
            batch_metrics = self.model.get_train_metrics(data_dict, predictions)
            
            # store data by recorder
            for name, value in batch_metrics.items():
                train_recorder_metric[name].update(value)
            for name, value in losses.items():
                train_recorder_loss[name].update(value.detach())
            
            if iteration % 300 == 0:
                # info for loss
                loss_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_loss.items():
                    loss_str += f"training-loss, {k}: {v.average():.3f}    "
                self.logger.info(loss_str)
                
                # info for metric
                metric_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_metric.items():
                    metric_str += f"training-metric, {k}: {v.average():.3f}    "
                self.logger.info(metric_str)

                # tensorboard updates
                for k, v in train_recorder_loss.items():
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_loss/{k}', v.average(), global_step=step_cnt)
                for k, v in train_recorder_metric.items():
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_metric/{k}', v.average(), global_step=step_cnt)
                
                # clear recorder for next block
                for name, recorder in train_recorder_loss.items():
                    recorder.clear()
                for name, recorder in train_recorder_metric.items():
                    recorder.clear()

            step_cnt += 1
            del data_dict
            del predictions
            del losses
            del batch_metrics
        
        # --- End of Iteration Loop: Run Test ---
        test_best_metric = {}
        if test_data_loaders is not None:
            self.logger.info("===> Test start!")
            test_best_metric = self.test_epoch(
                epoch, 
                iteration,
                test_data_loaders, 
                step_cnt,
            )

        self.logger.info("===> Epoch[{}] end with testing auc: {:.4f}!".format(epoch, test_best_metric.get('auc', 0.0)))    

        save_dir = os.path.join(self.log_dir, 'test', 'LibriSeVoc')
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"ckpt_{epoch}.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        if self.config['ngpu'] > 1:
            torch.save(self.model.module.state_dict(), save_path)
        else:
            torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Checkpoint saved to {save_path}")

        return test_best_metric
    
    def test_one_dataset(self, data_loader):
        test_recorder_loss = defaultdict(Recorder)
        for i, data_dict in tqdm(enumerate(data_loader)):
            data, label = data_dict['audio'], data_dict['label']
            label = torch.where(data_dict['label']!=0, 1, 0)
            if 'label_spe' in data_dict:
                data_dict.pop('label_spe')
        
            data_dict['audio'], data_dict['label'] = data.to(device), label.to(device)

            # Model forward during inference
            predictions = self.inference(data_dict)
            
            # compute all losses
            losses = self.model.get_losses(data_dict, predictions, inference=True)

            # store data by recorder
            for name, value in losses.items():
                test_recorder_loss[name].update(value.detach())

            del losses
            del data_dict
            del data
            del label
            
        return test_recorder_loss

    def test_epoch(self, epoch, iteration, test_data_loaders, step):
        self.setEval()

        losses_all_datasets = {}
        metrics_all_datasets = {}
        best_metrics_per_dataset = defaultdict(dict)
        
        keys = test_data_loaders.keys()
        for key in keys:
            # save testing metadata
            data_dict_meta = test_data_loaders[key].dataset.data_dict
            self.save_data_dict('test', data_dict_meta, key)

            # compute loss and metrics
            losses_one_dataset_recorder = self.test_one_dataset(test_data_loaders[key])
            metric_one_dataset = self.model.get_test_metrics()
            
            losses_all_datasets[key] = losses_one_dataset_recorder
            metrics_all_datasets[key] = metric_one_dataset
            
            # maintain the best metric and save
            best_metric = self.best_metrics_all_time[key].get(self.metric_scoring, float('-inf') if self.metric_scoring != 'eer' else float('inf'))
            improved = (metric_one_dataset[self.metric_scoring] > best_metric) if self.metric_scoring != 'eer' else (metric_one_dataset[self.metric_scoring] < best_metric)
            
            if improved:
                self.best_metrics_all_time[key][self.metric_scoring] = metric_one_dataset[self.metric_scoring]
                if self.config['save_ckpt']:
                    self.save_ckpt('test', key)
                # If we need to save the specific metrics dictionary
                self.save_metrics('test', metric_one_dataset, key)
        
            # Logging results for this dataset
            loss_str = f"dataset: {key}    step: {step}    "
            for k, v in losses_one_dataset_recorder.items():
                v_val = v.average().item() if isinstance(v.average(), torch.Tensor) else v.average()
                loss_str += f"testing-loss, {k}: {v_val:.4f}    "
            self.logger.info(loss_str)

            metric_str = f"dataset: {key}    step: {step}    "
            for k, v in metric_one_dataset.items():
                if k in ['pred', 'label']: continue # skip large arrays
                v_val = v.item() if isinstance(v, torch.Tensor) else v
                metric_str += f"testing-metric, {k}: {v_val:.4f}    "
            self.logger.info(metric_str)

            # Tensorboard reporting
            for k, v in losses_one_dataset_recorder.items():
                writer = self.get_writer('test', key, k)
                v_val = v.average().item() if isinstance(v.average(), torch.Tensor) else v.average()
                writer.add_scalar(f'test_losses/{k}', v_val, global_step=step)
            for k, v in metric_one_dataset.items():
                if k in ['pred', 'label']: continue
                writer = self.get_writer('test', key, k)
                v_val = v.item() if isinstance(v, torch.Tensor) else v
                writer.add_scalar(f'test_metrics/{k}', v_val, global_step=step)

        self.logger.info('===> Test Done!')
        
        # Return the specific scoring metric for the summary log in train.py
        summary_metrics = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in metrics_all_datasets[list(keys)[0]].items() if k not in ['pred', 'label']}
        return summary_metrics

    @torch.no_grad()
    def inference(self, data_dict):
        predictions = self.model(data_dict, inference=True)
        predictions = {k:v.detach() for k,v in predictions.items()}
        return predictions