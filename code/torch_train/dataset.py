from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import os
data_dir = '/mnt/sda/zzl/Dataset_clean_all_newfps'
class trainSet(Dataset):
    def __init__(self):
        #定义好 image 的路径
        self.input = np.load(os.path.join(data_dir,'train_data.npy'))
        print('1')
        self.input = np.transpose(self.input,(0,3,1,2))
        print('2')
        self.target = np.load(os.path.join(data_dir,'train_label_var.npy'))
        print('3')
        '''
        self.input = np.ones([560,1,32,64])
        self.target = np.ones([560, 61])
        '''
    def __getitem__(self, index):
        input = self.input[index]
        target = self.target[index]
        return input, target

    def __len__(self):
        return self.input.shape[0]