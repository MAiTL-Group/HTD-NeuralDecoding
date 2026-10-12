# Copyright 2026 The HTD Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.nn import functional as F
import pytorch_lightning as pl
import numpy as np
from torch import nn
import datetime
import os
import math
from sklearn.metrics import average_precision_score, roc_auc_score, hamming_loss
from pytorch_lightning.utilities import rank_zero_only

from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam, AdamW
import random

from .network import (
    BrainNetwork_mindeye_2d_light,BrainNetwork_mindeye_2d_fuse
)

def avg_list(num_list):
    return sum(num_list) / len(num_list)


def calculate_top_k_accuracy( logits, k):
    correct_topk = 0
    for i in range(logits.shape[0]):
        sorted_indices = torch.argsort(logits[i], descending=True)
        if i in sorted_indices[:k]:
            correct_topk += 1
    topk_accuracy = correct_topk / logits.shape[0]
    return topk_accuracy

def calculate_mAP_sk( logits, device):
    APs = []
    for i in range(logits.shape[0]):
        y_true = np.zeros(logits.shape[1])
        y_true[i] = 1
        y_scores = logits[i].detach().cpu().numpy()
        AP = average_precision_score(y_true, y_scores)
        APs.append(AP)
    mAP = np.mean(APs)
    return mAP

def calculate_auc( logits, device):
    aucs = []
    for i in range(logits.shape[0]):
        y_true = np.zeros(logits.shape[1])
        y_true[i] = 1
        y_scores = logits[i].detach().cpu().numpy()
        auc = roc_auc_score(y_true, y_scores)
        aucs.append(auc)
    return np.mean(aucs)

def calculate_hamming_loss( logits, device):
    y_true = np.zeros((logits.shape[0], logits.shape[1]))
    y_pred = np.zeros((logits.shape[0], logits.shape[1]))
    for i in range(logits.shape[0]):
        y_true[i, i] = 1
        y_pred[i] = (logits[i] > 0.5).detach().cpu().numpy()
    return hamming_loss(y_true, y_pred)

def load_brainnetwork(network,kargs):
    if network == 'BrainNetwork_mindeye_2d_fuse':
        brain_network = BrainNetwork_mindeye_2d_fuse(n_times=kargs.get("n_times",5), fuse_method=kargs.get("fuse_method","mlp"), normalization=kargs.get("normalization","bn"),activation=kargs.get("activation","relu"),projection=kargs.get("projection","nonlinear"))
    elif network == 'BrainNetwork_mindeye_2d_light':
        brain_network = BrainNetwork_mindeye_2d_light(n_times=kargs.get("n_times",5), normalization=kargs.get("normalization","bn"),activation=kargs.get("activation","relu"),projection=kargs.get("projection","nonlinear"))
    else:
        raise Exception("Network not found!")

    return brain_network


def mixco_nce(preds, targs, temp=0.1, perm=None, betas=None, select=None, distributed=False,
              accelerator=None, local_rank=None, bidirectional=True):
    brain_clip = (preds @ targs.T)/temp

    if perm is not None and betas is not None and select is not None:
        probs = torch.diag(betas)
        probs[torch.arange(preds.shape[0]).to(preds.device), perm] = 1 - betas

        loss = -(brain_clip.log_softmax(-1) * probs).sum(-1).mean()
        if bidirectional:
            loss2 = -(brain_clip.T.log_softmax(-1) * probs.T).sum(-1).mean()
            loss = (loss + loss2)/2
        return loss
    else:
        loss =  F.cross_entropy(brain_clip, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
        if bidirectional:
            loss2 = F.cross_entropy(brain_clip.T, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
            loss = (loss + loss2)/2
        return loss


def mixco(voxels, beta=0.15, s_thresh=0.5):
    perm = torch.randperm(voxels.shape[0])
    voxels_shuffle = voxels[perm].to(voxels.device,dtype=voxels.dtype)
    betas = torch.distributions.Beta(beta, beta).sample([voxels.shape[0]]).to(voxels.device,dtype=voxels.dtype)
    select = (torch.rand(voxels.shape[0]) <= s_thresh).to(voxels.device)
    betas_shape = [-1] + [1]*(len(voxels.shape)-1)
    voxels[select] = voxels[select] * betas[select].reshape(*betas_shape) + \
        voxels_shuffle[select] * (1 - betas[select]).reshape(*betas_shape)
    betas[~select] = 1
    return voxels, perm, betas, select

def soft_clip_loss(preds, targs, temp=0.125):
    clip_clip = (targs @ targs.T)/temp
    brain_clip = (preds @ targs.T)/temp

    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()

    loss = (loss1 + loss2)/2
    return loss

def cosine_anneal(start, end, steps):
    return end + (start - end)/2 * (1 + torch.cos(torch.pi*torch.arange(steps)/(steps-1)))



class Brain_Model(pl.LightningModule):
    def __init__(self, **kargs):
        super().__init__()
        self.save_hyperparameters()

        self.kargs=kargs

        self.loss_log=0
        self.brain_network=load_brainnetwork(kargs['network'],kargs)

        self.soft_loss_temps= cosine_anneal(0.004, 0.0075, self.kargs['max_epoch'] - int(0.33 * self.kargs['max_epoch']))
        self.soft_loss_temps_full= cosine_anneal(0.004, 0.0075, self.kargs['max_epoch'])
        self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052)  # from clip vit/L-14 pretrained model's logit_scale

        os.makedirs(self.kargs['default_root_dir'], exist_ok=True)

        self.log_file_path = self.kargs['default_root_dir']+f"/{self.kargs['exp_name']}_log_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        self.log_file = None
        self.setup_log_file()

        self.test_fmri_embeddings = []
        self.test_img_embeddings = []
        self.test_fmri_paths = []
        self.test_img_paths = []
        self.mse_loss = torch.nn.MSELoss()
        try:
            for subj in self.kargs['val_subjs']:
                self.test_fmri_embeddings.append([])
                self.test_img_embeddings.append([])
                self.test_fmri_paths.append([])
                self.test_img_paths.append([])
        except:
            for subj in self.kargs['test_subjs']:
                self.test_fmri_embeddings.append([])
                self.test_img_embeddings.append([])
                self.test_fmri_paths.append([])
                self.test_img_paths.append([])

        self.HCP_top1_accuracy_list_indis=[]
        self.HCP_top3_accuracy_list_indis=[]
        self.HCP_top5_accuracy_list_indis=[]
        self.HCP_top10_accuracy_list_indis=[]
        self.HCP_mAP_list_indis=[]
        self.HCP_AUC_list_indis=[]
        self.HCP_Hamming_list_indis=[]

        self.HCP_top1_accuracy_list_outdis=[]
        self.HCP_top3_accuracy_list_outdis=[]
        self.HCP_top5_accuracy_list_outdis=[]
        self.HCP_top10_accuracy_list_outdis=[]
        self.HCP_mAP_list_outdis=[]
        self.HCP_AUC_list_outdis=[]
        self.HCP_Hamming_list_outdis=[]

    @rank_zero_only
    def setup_log_file(self):
        self.log_file = open(self.log_file_path, 'w')

    def configure_optimizers(self):
        if self.kargs['lr_scheduler'] == 'OneCycleLR':
            if self.kargs['optimizer'] == 'Adam':
                optimizer = Adam(self.brain_network.parameters(), lr=float(float(self.kargs['lr'])/25), weight_decay=float(self.kargs['weight_decay']))
            elif self.kargs['optimizer'] == 'AdamW' :
                optimizer = AdamW(self.brain_network.parameters(), lr=float(float(self.kargs['lr'])/25), weight_decay=float(self.kargs['weight_decay']))
        else:
            if self.kargs['optimizer'] == 'Adam':
                optimizer = Adam(self.brain_network.parameters(), lr=float(self.kargs['lr']), weight_decay=float(self.kargs['weight_decay']))
            elif self.kargs['optimizer'] == 'AdamW' :
                optimizer = AdamW(self.brain_network.parameters(), lr=float(self.kargs['lr']), weight_decay=float(self.kargs['weight_decay']))


        if self.kargs['lr_scheduler'] == 'OneCycleLR':
                    ############################################################################################################
            with open(self.kargs['train_data_list']) as fin:
                if self.kargs['dataset_type'] =='NSD':
                    self.path_list = [line.strip() for line in fin if any(subject in line for subject in self.kargs['train_subjs'])]
                else:
                    self.path_list = [line.strip() for line in fin]

            print('Train_Data')
            train_num=len(self.path_list)
            print(len(self.path_list))

            steps_per_epoch=math.ceil(train_num/self.kargs['batch_size'])
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=float(self.kargs['lr']),
                steps_per_epoch=steps_per_epoch,
                epochs=self.kargs['max_epoch'],
                pct_start=0.1,
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=1e4,
            )

            return [optimizer]

        else:
            scheduler = StepLR(optimizer, step_size=int(self.kargs['max_epoch']/4), gamma=0.5)

            return [optimizer],[scheduler]

    def calc_sim(self, image_features, text_features):
        image_features=image_features.view(image_features.size(0),-1)
        text_features=text_features.view(text_features.size(0),-1)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_fmri = logits_per_image.t()
        return logits_per_image, logits_per_fmri

    def forward(self, fmri, img):
        _,fmri_features = self.brain_network.encode_fmri(fmri)
        image_features=img
        logits_per_image, logits_per_fmri = self.calc_sim(image_features, fmri_features)

        return image_features,fmri_features,logits_per_image, logits_per_fmri, {}


    def training_step(self, batch, batch_idx):
        loss=0
        fmri, img, subj, movie_id, frame_id = batch

        if self.kargs['loss_type']=='CLIPLoss':
            img_features,fmri_features,logits_per_img, logits_per_fmri, other_returns = self(fmri, img)

            labels_gt = torch.arange(logits_per_img.shape[0], dtype=torch.long).to(self.device)

            contrastive_loss = (
                F.cross_entropy(logits_per_img, labels_gt) +
                F.cross_entropy(logits_per_fmri, labels_gt)
            ) / 2

            loss=loss+contrastive_loss

        elif self.kargs['loss_type']=='SoftCLIPLoss':
            img_features,fmri_features,logits_per_img, logits_per_fmri, other_returns = self(fmri, img)
            clip_voxels_norm = nn.functional.normalize(fmri_features.flatten(1), dim=-1)
            clip_target_norm = nn.functional.normalize(img_features.flatten(1), dim=-1)

            epoch_temp = self.soft_loss_temps_full[self.current_epoch - int(0.33*self.kargs['max_epoch'])]
            loss_nce = soft_clip_loss(
                clip_voxels_norm,
                clip_target_norm,
                temp=epoch_temp)
            loss=loss+loss_nce

        elif self.kargs['loss_type']=='BiMixCo_SoftCLIP':

            if self.current_epoch < int(0.33 * self.kargs['max_epoch']):
                fmri, perm, betas, select = mixco(fmri)

            img_features,fmri_features,logits_per_img, logits_per_fmri, other_returns = self(fmri, img)
            clip_voxels_norm = nn.functional.normalize(fmri_features.flatten(1), dim=-1)
            clip_target_norm = nn.functional.normalize(img_features.flatten(1), dim=-1)


            if self.current_epoch < int(0.33 * self.kargs['max_epoch']):
                loss_nce = mixco_nce(
                    clip_voxels_norm,
                    clip_target_norm,
                    temp=.006,
                    perm=perm, betas=betas, select=select)
                loss=loss+loss_nce
            else:
                epoch_temp = self.soft_loss_temps[self.current_epoch - int(0.33*self.kargs['max_epoch'])]
                loss_nce = soft_clip_loss(
                    clip_voxels_norm,
                    clip_target_norm,
                    temp=epoch_temp)
                loss=loss+loss_nce

        elif self.kargs['loss_type']=='MSE':

            img_features,fmri_features,logits_per_img, logits_per_fmri, other_returns = self(fmri, img)
            mse_loss=self.mse_loss(img_features,fmri_features)
            loss=loss+mse_loss

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=len(batch),sync_dist=True)

        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr, on_step=True, on_epoch=False, prog_bar=True, logger=True,batch_size=len(batch))

        self.loss_log = loss
        torch.cuda.empty_cache()

        return loss

    @rank_zero_only
    def on_train_start(self) -> None:

        self.log_file.write("Hyperparameters:\n")
        for key, value in self.hparams.items():
            self.log_file.write(f"{key}: {value}\n")
        self.log_file.write("\n\n")

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total_params = trainable_params + non_trainable_params

        params_info = f"""
        ---------------------------------------------------------
        Trainable params: {int(trainable_params/1000000)}M
        Non-trainable params: {int(non_trainable_params/1000000)}M
        Total params: {int(total_params/1000000)}M
        ---------------------------------------------------------
        """

        self.log_file.write("Model Summary:\n")
        self.log_file.write(params_info)
        self.log_file.write(self.model_summary())
        self.log_file.write("\n\n")
        self.log_file.flush()


    @rank_zero_only
    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0) -> None:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lr = self.trainer.optimizers[0].param_groups[0]['lr']

        if self.kargs['lr_scheduler'] == 'OneCycleLR':
            self.scheduler.step()

        self.log_file.write(f'Epoch: {self.current_epoch}, Batch: {batch_idx}, Time: {current_time}, Loss: {self.loss_log.item()}, LR: {lr}\n')
        self.log_file.flush()

    @rank_zero_only
    def on_train_end(self) -> None:
        self.log_file.close()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss=0
        id_=0
        for subj in self.kargs['val_subjs']:

            if dataloader_idx == id_:

                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                fmri, img, subj, movie_id, frame_id = batch
                img_features,fmri_features,logits_per_img, logits_per_fmri, other_returns = self(fmri, img)

                self.test_fmri_embeddings[id_].append(fmri_features)
                self.test_img_embeddings[id_].append(img_features)

                self.test_img_paths[id_].append("")
                self.test_fmri_paths[id_].append("")

            id_+=1


        torch.cuda.empty_cache()

        return loss


    def on_validation_epoch_end(self):

        id_=0
        for subj in self.kargs['val_subjs']:

            # merge batches
            all_fmri_embeddings_ = torch.cat(self.test_fmri_embeddings[id_],dim=0)
            all_img_embeddings_ = torch.cat(self.test_img_embeddings[id_],dim=0)

            # to main GPU card
            all_fmri_embeddings_ = self.all_gather(all_fmri_embeddings_)
            all_img_embeddings_ = self.all_gather(all_img_embeddings_)

            # combine data from different GPU
            all_fmri_embeddings_ = torch.cat([x for x in all_fmri_embeddings_], dim=0)
            all_img_embeddings_ = torch.cat([x for x in all_img_embeddings_], dim=0)

            if self.trainer.global_rank == 0:

                all_fmri_embeddings = all_fmri_embeddings_
                all_img_embeddings = all_img_embeddings_

                print('all_fmri_embeddings',all_fmri_embeddings.shape)

                logits_per_img, logits_per_fmri = self.calc_sim(all_img_embeddings.float(),all_fmri_embeddings.float())

                logits_per_fmri_sigmod=torch.sigmoid(logits_per_fmri)

                top1_accuracy = calculate_top_k_accuracy(logits_per_fmri, k=1)
                top3_accuracy = calculate_top_k_accuracy(logits_per_fmri, k=3)
                top5_accuracy = calculate_top_k_accuracy(logits_per_fmri, k=5)
                top10_accuracy = calculate_top_k_accuracy(logits_per_fmri, k=10)

                mAP = calculate_mAP_sk(logits_per_fmri, device=self.device)

                AUC = calculate_auc(logits_per_fmri_sigmod, device=self.device)
                Hamming = calculate_hamming_loss(logits_per_fmri_sigmod, device=self.device)

                self.log_file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"+"\n")
                self.log_file.write("Testing on "+subj+"\n")
                self.log_file.write(f'test_top1_accuracy: {top1_accuracy} \n')
                self.log_file.write(f'test_top3_accuracy: {top3_accuracy} \n')
                self.log_file.write(f'test_top5_accuracy: {top5_accuracy} \n')
                self.log_file.write(f'test_top10_accuracy: {top10_accuracy} \n')
                self.log_file.write(f'test_mAP: {mAP} \n')
                self.log_file.write(f'test_AUC: {AUC} \n')
                self.log_file.write(f'test_Hamming: {Hamming} \n')
                self.log_file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"+"\n")

                self.log_file.flush()

                print("Testing on "+subj)
                print('test_top1_accuracy', top1_accuracy)
                print('test_top3_accuracy', top3_accuracy)
                print('test_top5_accuracy', top5_accuracy)
                print('test_top10_accuracy', top10_accuracy)
                print('test_mAP', mAP)
                print('test_AUC', AUC)
                print('test_Hamming', Hamming)

            self.test_fmri_embeddings[id_] = []
            self.test_img_embeddings[id_] = []

            self.test_img_paths[id_]=[]
            self.test_fmri_paths[id_]=[]

            id_+=1

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        id_=0
        for subj in self.kargs['test_subjs']:

            if dataloader_idx == id_:

                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                fmri, img, _, movie_id, frame_id = batch
                img_features,fmri_features,logits_per_img, logits_per_fmri, other_returns = self(fmri, img)

                logits_per_img, logits_per_fmri = self.calc_sim(img_features.float(),fmri_features.float())

                logits_per_fmri_sigmod=torch.sigmoid(logits_per_fmri)

                top1_accuracy = calculate_top_k_accuracy(logits_per_fmri, k=1)
                top3_accuracy = calculate_top_k_accuracy(logits_per_fmri, k=3)
                top5_accuracy = calculate_top_k_accuracy(logits_per_fmri, k=5)
                top10_accuracy = calculate_top_k_accuracy(logits_per_fmri, k=10)

                mAP = calculate_mAP_sk(logits_per_fmri, device=self.device)

                AUC = calculate_auc(logits_per_fmri_sigmod, device=self.device)
                Hamming = calculate_hamming_loss(logits_per_fmri_sigmod, device=self.device)

                self.log_file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"+"\n")
                self.log_file.write("Testing on "+subj+"\n")
                self.log_file.write(f'test_top1_accuracy: {top1_accuracy} \n')
                self.log_file.write(f'test_top3_accuracy: {top3_accuracy} \n')
                self.log_file.write(f'test_top5_accuracy: {top5_accuracy} \n')
                self.log_file.write(f'test_top10_accuracy: {top10_accuracy} \n')
                self.log_file.write(f'test_mAP: {mAP} \n')
                self.log_file.write(f'test_AUC: {AUC} \n')
                self.log_file.write(f'test_Hamming: {Hamming} \n')
                self.log_file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"+"\n")

                self.log_file.flush()

                print("Testing on "+subj)
                print('test_top1_accuracy', top1_accuracy)
                print('test_top3_accuracy', top3_accuracy)
                print('test_top5_accuracy', top5_accuracy)
                print('test_top10_accuracy', top10_accuracy)
                print('test_mAP', mAP)
                print('test_AUC', AUC)
                print('test_Hamming', Hamming)

                if subj in ["100610_Subj1","102311_Subj2","102816_Subj3","104416_Subj4","105923_Subj5","108323_Subj6","109123_Subj7","111312_Subj8","111514_Subj9","114823_Subj10"]:
                    self.HCP_top1_accuracy_list_outdis.append(top1_accuracy)
                    self.HCP_top3_accuracy_list_outdis.append(top3_accuracy)
                    self.HCP_top5_accuracy_list_outdis.append(top5_accuracy)
                    self.HCP_top10_accuracy_list_outdis.append(top10_accuracy)
                    self.HCP_mAP_list_outdis.append(mAP)
                    self.HCP_AUC_list_outdis.append(AUC)
                    self.HCP_Hamming_list_outdis.append(Hamming)
                else:
                    self.HCP_top1_accuracy_list_indis.append(top1_accuracy)
                    self.HCP_top3_accuracy_list_indis.append(top3_accuracy)
                    self.HCP_top5_accuracy_list_indis.append(top5_accuracy)
                    self.HCP_top10_accuracy_list_indis.append(top10_accuracy)
                    self.HCP_mAP_list_indis.append(mAP)
                    self.HCP_AUC_list_indis.append(AUC)
                    self.HCP_Hamming_list_indis.append(Hamming)

            id_+=1

        torch.cuda.empty_cache()

    def on_test_epoch_end(self):

        if self.kargs['dataset_type'] == "HCP":
            self.log_file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"+"\n")
            self.log_file.write("Final Average Results on outdis (1-10) "+"\n")
            self.log_file.write(f'test_top1_accuracy: {avg_list(self.HCP_top1_accuracy_list_outdis):.2f} \n')
            self.log_file.write(f'test_top3_accuracy: {avg_list(self.HCP_top3_accuracy_list_outdis):.2f} \n')
            self.log_file.write(f'test_top5_accuracy: {avg_list(self.HCP_top5_accuracy_list_outdis):.2f} \n')
            self.log_file.write(f'test_top10_accuracy: {avg_list(self.HCP_top10_accuracy_list_outdis):.2f} \n')
            self.log_file.write(f'test_mAP: {avg_list(self.HCP_mAP_list_outdis)} \n')
            self.log_file.write(f'test_AUC: {avg_list(self.HCP_AUC_list_outdis)} \n')
            self.log_file.write(f'test_Hamming: {avg_list(self.HCP_Hamming_list_outdis)} \n')
            self.log_file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"+"\n\n")

            self.log_file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"+"\n")
            self.log_file.write("Final Average Results on indis (11-177) "+"\n")
            self.log_file.write(f'test_top1_accuracy: {avg_list(self.HCP_top1_accuracy_list_indis):.2f} \n')
            self.log_file.write(f'test_top3_accuracy: {avg_list(self.HCP_top3_accuracy_list_indis):.2f} \n')
            self.log_file.write(f'test_top5_accuracy: {avg_list(self.HCP_top5_accuracy_list_indis):.2f} \n')
            self.log_file.write(f'test_top10_accuracy: {avg_list(self.HCP_top10_accuracy_list_indis):.2f} \n')
            self.log_file.write(f'test_mAP: {avg_list(self.HCP_mAP_list_indis)} \n')
            self.log_file.write(f'test_AUC: {avg_list(self.HCP_AUC_list_indis)} \n')
            self.log_file.write(f'test_Hamming: {avg_list(self.HCP_Hamming_list_indis)} \n')
            self.log_file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"+"\n\n")


    def model_summary(self):
        return str(self)
