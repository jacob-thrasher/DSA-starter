import os
import random
import pickle
import scipy
import torch
import numpy as np
import pandas as pd
import torchio as tio
import torchvision.transforms as T
import nibabel as nib
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 
from pycox.datasets import metabric, support, gbsg, flchain
from torch.utils.data import Dataset

def split_df(df):
    df_test = df.sample(frac=0.2)
    df.drop(df_test.index)
    df_valid = df.sample(frac=0.2)
    df.drop(df_valid.index)

    return df, df_valid, df_test

def centerCrop(img, length, width, height):
    assert img.shape[1] >= length
    assert img.shape[2] >= width
    # assert img.shape[3] >= height

    # T2 MRIs only have 42 slices
    if img.shape[3] < height:
        transform = tio.transforms.Pad((length, width, height))
        img = transform(img)

    x = img.shape[1]//2 - length//2
    y = img.shape[2]//2 - width//2
    z = img.shape[3]//2 - height//2
    img = img[:,x:x+length, y:y+width, z:z+height]
    return img

def randomCrop(img, length, width, height):
    assert img.shape[1] >= length
    assert img.shape[2] >= width
    # assert img.shape[3] >= height

    # T2 MRIs only have 42 slices
    if img.shape[3] < height:
        transform = tio.transforms.Pad((length, width, height))
        img = transform(img)


    x = random.randint(0, img.shape[1] - length)
    y = random.randint(0, img.shape[2] - width)
    z = random.randint(0, img.shape[3] - height )
    img = img[:,x:x+length, y:y+width, z:z+height]
    return img



def load_dataset(root, dataset, n_bins=10):
    '''
    Loads and converts desired dataset to a survival dataset

    Args:
        root - Path to cached dataset
        dataset - name of dataset to load
    
    Keyword args:
        dim    - Desired dimension
        n_bins - Desired number of bins to discretize continuous data (Default: 10)
    '''


    if 'ADNI' in dataset:
        train_surv = ADNI_3D(dir_to_scans=os.path.join(root, f'ADNI/ADNI-T1/mni'), dir_to_tsv=os.path.join(root, f'ADNI/ADNI-T1/tabular_data'), dir_type='bids', 
                             mode='Train', n_label=2, name=dataset)
        valid_surv = ADNI_3D(dir_to_scans=os.path.join(root, f'ADNI/ADNI-T1/mni'), dir_to_tsv=os.path.join(root, f'ADNI/ADNI-T1/tabular_data'), dir_type='bids', 
                             mode='Val', n_label=2, name=dataset)
        test_surv  = ADNI_3D(dir_to_scans=os.path.join(root, f'ADNI/ADNI-T1/mni'), dir_to_tsv=os.path.join(root, f'ADNI/ADNI-T1/tabular_data'), dir_type='bids', 
                             mode='Test', n_label=2, name=dataset)
        time_steps = np.arange(0, 11, 1) # ADNI data will be loaded as [0, 60] in 6mo increments, and normalized to [0, 10]

    elif dataset in ['METABRIC', 'SUPPORT', 'GBSG', 'FLCHAIN']:
        if dataset == 'METABRIC': 
            df = metabric.read_df()
            cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
            cols_leave = ['x4', 'x5', 'x6', 'x7']

        elif dataset == 'SUPPORT': # 14
            df = support.read_df()
            cols_standardize = ['x0', 'x2', 'x3', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']
            cols_leave = ['x1', 'x4', 'x5']

        elif dataset == 'GBSG':
            df = gbsg.read_df()
            cols_standardize = ['x1', 'x3', 'x4', 'x5', 'x6']
            cols_leave = ['x0', 'x2']

        elif dataset == 'FLCHAIN':
            # Note: Will raise error in pycox==0.2.3 --> I edited the source code to remove Unnamed:0 from drop list
            df = flchain.read_df()
            df.drop(columns=['rownames'], inplace=True)
            cols_standardize = ['age', 'kappa', 'lambda', 'flc.grp', 'creatinine']
            cols_leave = ['sex', 'mgus']
            df.rename(columns={'futime': 'duration', 'death': 'event'}, inplace=True)

        df_train, df_valid, df_test = split_df(df)

        # Standardize features
        standardize = [([col], StandardScaler()) for col in cols_standardize]
        leave = [(col, None) for col in cols_leave]
        x_mapper = DataFrameMapper(standardize + leave)
        x_train = x_mapper.fit_transform(df_train).astype('float32')
        x_valid = x_mapper.transform(df_valid).astype('float32')
        x_test = x_mapper.transform(df_test).astype('float32')

        labtrans = LabTransDiscreteTime(n_bins)
        get_target = lambda df: (df['duration'].values, df['event'].values)
        t_train, e_train = labtrans.fit_transform(*get_target(df_train))
        t_valid, e_valid = labtrans.transform(*get_target(df_valid))
        t_test, e_test = labtrans.transform(*get_target(df_test))


        train_surv = PyCoxDataset(x_train, t_train, e_train, name=dataset)
        valid_surv = PyCoxDataset(x_valid, t_valid, e_valid, name=dataset)
        test_surv = PyCoxDataset(x_test, t_test, e_test, name=dataset)
        time_steps = np.arange(0, n_bins, 1)

        
    return train_surv, valid_surv, test_surv, time_steps



class PyCoxDataset(Dataset):
    def __init__(self, features, times, events, name='Dataset'):

        self.features = features
        self.times = times
        self.events = events
        self.name = name

    
    def __len__(self):
        return len(self.times)
    
    def __getitem__(self, idx): 
        X = self.features[idx]
        t = self.times[idx]
        e = self.events[idx]

        return X, t, e, -1


class ADNI_3D(Dataset):

    def __init__(self, 
                 dir_to_scans, 
                 dir_to_tsv, 
                 dir_type='bids', 
                 mode='Train', 
                 n_label=3, 
                 label_type='future',
                 name='ADNI',
                 mri_weight='T1',
                 isOOD=False):
        '''
        Args:
            dir_to_sans - path to folder containing MRI scans in BIDS format
            dir_to_tsv - path to tsv file (not sure what info this contains yet)
            dir_type - directory structure of MRI scans.
                        Value must be in ['bids', 'caps']
                        See clinica documentation for more information: 
                        https://aramislab.paris.inria.fr/clinica/docs/public/latest/
            mode - Train/test mode
            n_label - number of labels to include
                        n_label=2 --> CN and AD only
                        n_label=3 --> CN, MCI, AD
            percentage_usage - Percentage of data to use. This is for defining the size of the training set compared to test
        '''
        assert dir_type in ['bids', 'caps'], f'param "dir_type" must be in [bids, caps], got {dir_type}'

        # NOTE: "Stubby" and "Past" loaders not yet implemented
        assert label_type in ['future'], f'param "label_type" must be in [future], got {label_type}'

        # Define label mapping -- Don't need this, but won't delete yet
        if n_label == 3:
            LABEL_MAPPING = ["CN", "MCI", "AD"]
        elif n_label == 2:
            LABEL_MAPPING = ["CN", "AD"]
        self.LABEL_MAPPING = LABEL_MAPPING  
        self.mode = mode
        self.name = name
        self.isOOD = isOOD
        self.mri_weight = mri_weight
        subject_tsv = pd.io.parsers.read_csv(os.path.join(dir_to_tsv, f'{mode}_survival.csv'))

        # Clean sessions without labels
        indices_not_missing = []
        for i in range(len(subject_tsv)):
            if mode == 'Train':
                if (subject_tsv.iloc[i].diagnosis in LABEL_MAPPING):
                    indices_not_missing.append(i)
            else:
                if (subject_tsv.iloc[i].diagnosis in LABEL_MAPPING):
                    indices_not_missing.append(i)

        self.subject_tsv = subject_tsv.iloc[indices_not_missing]


        # Get all subject IDs from dataset
        self.subject_id = np.unique(subject_tsv.participant_id.values)
        self.dir_to_scans = dir_to_scans
        self.label_type = label_type
        self.dir_type = dir_type
        self.mode = mode
        self.n_label = n_label
     

    
    # def augment_image(self, image):
    #     sigma = np.random.uniform(0.0,1.0,1)[0]
    #     image = scipy.ndimage.filters.gaussian_filter(image, sigma, truncate=8)
    #     return image

    def __len__(self):
        return len(self.subject_tsv)

    def __getitem__(self, idx):
        '''
        Returns:
            X     - Features
            t     - time of event/censor
            e     - Event indicator
            label - AD classifiction label (0: CN/MCI, 1: AD)
        '''
        row = self.subject_tsv.iloc[idx]
        # ------------------------
        # CLASSIFIER CODE
        dx = row.diagnosis
        if self.n_label == 2:
            if dx == 'CN' or dx == 'MCI':
                label = 0
            elif dx == 'AD':
                label = 1
            else:
                label = -100 # Missing labels
        elif self.n_label == 3:
            if dx == 'CN': label = 0
            elif dx == 'MCI': label = 1
            elif dx == 'AD': label = 2
            else: label = -100
        

        # --------------------------------
        # SURVIVAL ANALYSIS DATA

        # Get event indicator
        # Doing this in __getitem__ to ensure alignment between image and labels
        t = row['time']
        e = row['event']


        # --------------------------------
        # LOAD MRI
                        
        path = os.path.join(self.dir_to_scans, self.subject_tsv.iloc[idx].participant_id,
                self.subject_tsv.iloc[idx].session_id,'anat') 
        all_segs = list(os.listdir(path))

        try:
            for seg_name in all_segs:
                # if 'Space_T1w' in seg_name or self.dir_type == 'bids':
                if 'nii.gz' in seg_name:
                    # image = nib.load(os.path.join(path,seg_name)).get_fdata().squeeze()
                    image = tio.ScalarImage(os.path.join(path, seg_name)).numpy()
                    if self.mode == 'Train': image = randomCrop(image, 96, 96, 96)
                    else: image = centerCrop(image, 96, 96, 96)

        except Exception as ex:
            print(f"Failed to load #{idx}: {path}")
            raise ex


        if not self.isOOD:
            return torch.tensor(image.astype(np.float32)), int(t), int(e), label
        else:
            image = self.augment(torch.tensor(image.astype(np.float32)))
            return image, int(t), int(e), label