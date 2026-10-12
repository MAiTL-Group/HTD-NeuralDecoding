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

import random
import torch
import os
import numpy as np
import torch.utils.data as data

from functools import partial

class FeatureManager:
    cache={}
    @classmethod
    def get_feature(cls, movie_id, frame_id, feature_type):
        movie_files = ['7T_MOVIE1_CC1_v2.mp4', '7T_MOVIE2_HO1_v2.mp4', '7T_MOVIE3_CC2_v2.mp4', '7T_MOVIE4_HO2_v2.mp4']
        if movie_id not in cls.cache:
            cls.cache[movie_id]=np.load(f"/opt/data/private/projects/workspaces/NeuralDecoding/HTD_src/dataset/HCP/preprocessed_stimulis/{movie_files[movie_id-1]}_{feature_type}.npy").astype(np.float32)
        return cls.cache[movie_id][frame_id]
    
class fMRIProcess:
    mask_roi=torch.tensor(np.load("/opt/data/private/projects/workspaces/NeuralDecoding/HTD_src/data/vc_masks.npy")!=0)
    @classmethod
    def average_5s(cls, arr, frame_id, delay=0, ntimes=5):
        return torch.from_numpy(arr[frame_id+delay:frame_id+ntimes+delay]).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).mean(dim=0,keepdim=True)*cls.mask_roi[None]
    @classmethod
    def concat_5s(cls, arr,frame_id, delay=0, ntimes=5):
        return torch.from_numpy(arr[frame_id+delay:frame_id+ntimes+delay]).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)*cls.mask_roi[None]


fmri_preprocess_func={
    "mix5s": fMRIProcess.average_5s,
    "concat1s_delay5": partial(fMRIProcess.concat_5s,ntimes=1,delay=5),
    "concat5s": fMRIProcess.concat_5s,
}


class Train_Data_HCP(data.Dataset):
    def __init__(self, train=True, **kargs):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.kargs = kargs

        self.check_files()

        self.file_objs={}
        for subj in self.kargs['train_subjs']:
            for movie in self.movies:
                npy_fname=os.path.join(f"{self.kargs['fmri_data_path']}",subj.split("_")[0],movie+"_image.npy")
                self.file_objs[npy_fname]=np.load(npy_fname)

        self.mask_roi=torch.tensor(np.load("/opt/data/private/projects/workspaces/NeuralDecoding/HTD_src/data/vc_masks.npy")!=0)

    def check_files(self):
        movies = ["MOVIE1","MOVIE2","MOVIE3","MOVIE4"]
        with open(self.kargs['train_data_list']) as fin:
            self.path_list = [line.strip() for line in fin]
            movie_lists=[movies[int(line[1])-1] for line in self.path_list]
        self.movies=list(set(movie_lists))
        print('Train_Data')
        print(len(self.path_list))

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        fmri_dirs=self.kargs['fmri_data_path']

        path = self.path_list[idx]
        filename = path
        movie_id=int(filename[1]);frame_id=int(filename.split("_")[1])

        img_feature = FeatureManager.get_feature(movie_id,frame_id,self.kargs["img_type"])

        subj=random.choice(self.kargs['train_subjs'])

        npy_filename=os.path.join(fmri_dirs,subj.split("_")[0],f"MOVIE{movie_id}_image.npy")
        fmri=fmri_preprocess_func[self.kargs["fmri_type"]](self.file_objs[npy_filename],frame_id)

        return fmri, img_feature, subj, movie_id, frame_id

class Val_Data_HCP(data.Dataset):
    def __init__(self, train=False, val_subj='',**kargs):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.kargs = kargs
        self.val_subj=val_subj

        self.check_files()

        self.file_objs={}
        for movie in self.movies:
            npy_fname=os.path.join(f"{self.kargs['fmri_data_path']}",self.val_subj.split("_")[0],movie+"_image.npy")
            self.file_objs[npy_fname]=np.load(npy_fname)
                
        self.mask_roi=torch.tensor(np.load("/opt/data/private/projects/workspaces/NeuralDecoding/HTD_src/data/vc_masks.npy")!=0)

    def check_files(self):
        movies = ["MOVIE1","MOVIE2","MOVIE3","MOVIE4"]
        with open(self.kargs['val_data_list']) as fin:
            self.path_list = [line.strip() for line in fin]
            movie_lists=[movies[int(line[1])-1] for line in self.path_list]
        self.movies=list(set(movie_lists))
        print('Val_Data')
        print(len(self.path_list))

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        fmri_dirs=self.kargs['fmri_data_path']

        path = self.path_list[idx]
        filename = path
        movie_id=int(filename[1]);frame_id=int(filename.split("_")[1])

        img_feature = FeatureManager.get_feature(movie_id,frame_id,self.kargs["img_type"])

        subj=self.val_subj
        npy_filename=os.path.join(fmri_dirs,subj.split("_")[0],f"MOVIE{movie_id}_image.npy")
        fmri=fmri_preprocess_func[self.kargs["fmri_type"]](self.file_objs[npy_filename],frame_id)

        return fmri, img_feature, subj, movie_id, frame_id

class Test_Data_HCP(data.Dataset):
    def __init__(self, train=False, test_subj='',**kargs):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.kargs = kargs
        self.test_subj=test_subj

        self.check_files()

        self.file_objs={}
        for movie in self.movies:
            npy_fname=os.path.join(f"{self.kargs['fmri_data_path']}",self.test_subj.split("_")[0],movie+"_image.npy")
            self.file_objs[npy_fname]=np.load(npy_fname)

        self.mask_roi=torch.tensor(np.load("/opt/data/private/projects/workspaces/NeuralDecoding/HTD_src/data/vc_masks.npy")!=0)

    def check_files(self):
        movies = ["MOVIE1","MOVIE2","MOVIE3","MOVIE4"]
        with open(self.kargs['test_data_list']) as fin:
            self.path_list = [line.strip() for line in fin]
            movie_lists=[movies[int(line[1])-1] for line in self.path_list]
        self.movies=list(set(movie_lists))
        print('Test_Data')
        print(len(self.path_list))

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        fmri_dirs=self.kargs['fmri_data_path']

        path = self.path_list[idx]
        filename = path
        movie_id=int(filename[1]);frame_id=int(filename.split("_")[1])

        img_feature = FeatureManager.get_feature(movie_id,frame_id,self.kargs["img_type"])

        subj=self.test_subj

        npy_filename=os.path.join(fmri_dirs,subj.split("_")[0],f"MOVIE{movie_id}_image.npy")
        fmri=fmri_preprocess_func[self.kargs["fmri_type"]](self.file_objs[npy_filename],frame_id)

        return fmri, img_feature, subj, movie_id, frame_id

