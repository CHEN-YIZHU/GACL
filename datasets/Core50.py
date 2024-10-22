import os
import glob
from torchvision.transforms import transforms
from typing import Callable, Optional
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
import tqdm
from shutil import move, rmtree
import zipfile


class CORe50(Dataset):
    def __init__(self,root             : str, 
                 train            : bool, 
                 transform        : Optional[Callable] = None, 
                 target_transform : Optional[Callable] = None, 
                 download         : bool = False):        
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        self.url = 'http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip'
        self.filename = 'core50_128x128.zip'

        self.fpath = os.path.join(root, 'core50_128x128')
        if not os.path.isfile(self.fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                download_url(self.url, root, filename=self.filename)

        if not os.path.exists(os.path.join(root, 'core50_128x128')):
            with zipfile.ZipFile(os.path.join(self.root, self.filename), 'r') as zf:
                for member in tqdm.tqdm(zf.infolist(), desc=f'Extracting {self.filename}'):
                    try:
                        zf.extract(member, root)
                    except zipfile.error as e:
                        pass

        self.train_session_list = ['s1', 's2', 's4', 's5', 's6', 's8', 's9', 's11']
        self.test_session_list = ['s3', 's7', 's10']
        self.label = [f'o{i}' for i in range(1, 51)]
        
        if not os.path.exists(self.fpath + '/train') and not os.path.exists(self.fpath + '/test'):
            self.split()
        
        if self.train:
            fpath = self.fpath + '/train'
        else:
            fpath = self.fpath + '/test'
        self.dataset = ImageFolder(fpath, transforms.ToTensor() if transform is None else transform, target_transform) 
        self.classes = [str(i) for i in range(50)]

        self.targets = []
        for i in self.dataset.targets:
            self.targets.append(i)
        pass
        
        


    def __getitem__(self, index):
        image, label = self.dataset.__getitem__(index)
        return image.expand(3,-1,-1), label
        
    def __len__(self):
        return len(self.dataset)
    

    def split(self):
        train_folder = self.fpath + '/train'
        test_folder = self.fpath + '/test'

        if os.path.exists(train_folder):
            rmtree(train_folder)
        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(train_folder)
        os.mkdir(test_folder)

        for s in tqdm.tqdm(self.train_session_list, desc='Preprocessing'):
            for l in self.label:
                dst = os.path.join(train_folder, l)
                if not os.path.exists(dst):
                    os.mkdir(os.path.join(train_folder, l))
                
                f = glob.glob(os.path.join(self.fpath, s, l, '*.png'))

                for src in f:
                    move(src, dst)
            rmtree(os.path.join(self.fpath, s))
        
        for s in tqdm.tqdm(self.test_session_list, desc='Preprocessing'):
            for l in self.label:
                dst = os.path.join(test_folder, l)

                if not os.path.exists(dst):
                    os.mkdir(os.path.join(test_folder, l))
                
                f = glob.glob(os.path.join(self.fpath, s, l, '*.png'))

                for src in f:
                    move(src, dst)
            rmtree(os.path.join(self.fpath, s))