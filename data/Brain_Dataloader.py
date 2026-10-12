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

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data.HCP_Datasets import Train_Data_HCP,Val_Data_HCP,Test_Data_HCP

class Brain_Dataloader(pl.LightningDataModule):

    def __init__(self, num_workers=0, stage=0,
                 **kargs):
        super().__init__()
        self.num_workers = num_workers
        self.kargs = kargs
        self.stage=stage
        self.setup(stage)


    def setup(self, stage=None):
        if self.kargs['dataset_type'] == "HCP":
            if stage == 'fit' or stage is None:
                # Assign train/val datasets for use in dataloaders
                self.trainset = Train_Data_HCP(train=True,**self.kargs)
                self.valsets={}
                for subj in self.kargs['val_subjs']:
                    valset = Val_Data_HCP(train=False,val_subj=subj,**self.kargs)
                    self.valsets[subj]=valset

            # Assign test dataset for use in dataloader(s)
            if stage == 'test' or stage is None:
                self.testsets={}
                for subj in self.kargs['test_subjs']:
                    testset = Test_Data_HCP(train=False,test_subj=subj,**self.kargs)
                    self.testsets[subj]=testset
        else:
            raise Exception("Wrong Dataset Type")


    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.kargs['batch_size'], num_workers=self.num_workers, shuffle=True,persistent_workers=True,pin_memory=True)

    def val_dataloader(self):
        return [DataLoader(self.valsets[subj], batch_size=self.kargs['batch_size'], num_workers=self.num_workers, shuffle=False) for subj in self.kargs['val_subjs']]

    def test_dataloader(self):
        return [DataLoader(self.testsets[subj], batch_size=self.kargs['batch_size'], num_workers=self.num_workers, shuffle=False) for subj in self.kargs['test_subjs']]
