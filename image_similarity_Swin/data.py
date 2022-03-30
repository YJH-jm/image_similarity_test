__all__ = ["TrainDataset"], 

import torch
from PIL import Image
import os
from torch.utils.data import Dataset
import glob
import config
import torchvision.transforms as transforms
import numpy as np

from model import ft_net_swin

# 한국인 재식별 이미지를 DataLoader로 사용하기 위한 작업

class TrainDataset(Dataset):
    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.image_dir = glob.glob(main_dir + "/**/*.png", recursive=True)
        self.classes_name = set()
        self.label = []
        
        for i in self.image_dir:
            tmp = i.split('\\')[-1]
            try:
                tmp = tmp.split('_')[1] + tmp.split('_')[2]
                self.label.append(tmp)
                self.classes_name.add(tmp)
            except:
                print(i)
        self.classes_name = list(self.classes_name)
        self.classes_name.sort()
        # print(self.classes_name)
        # print("class의 총 수 : ", len(self.classes_name))
        
        

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, idx): 
        img_loc = self.image_dir[idx]
        image = Image.open(img_loc).convert("RGB")
        target = self.label[idx]
        label = self.classes_name.index(target)
        if self.transform is not None:
            tensor_image = self.transform(image)
        return tensor_image, label


class TestDataset(Dataset):
    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.image_dir = glob.glob(main_dir + "/**/*.png", recursive=True) 
        self.classes_name = set()
        self.label = []

        for i in self.image_dir:
            # break
            i= i.split('\\')[2]            
            self.label.append(i)
            self.classes_name.add(i)
        self.classes_name = list(self.classes_name)
        self.classes_name.sort()
        # print(self.classes_name)
        # print("class의 총 수 : ", len(self.classes_name))
        
        

    def __len__(self): # 데이터셋의 길이, 즉 샘플의 수를 적어주는 부분
        return len(self.image_dir)

    def __getitem__(self, idx): # 데이터셋에서  특정 1개의 샘플을 가져오는 함수 
        img_loc = self.image_dir[idx]
        image = Image.open(img_loc).convert("RGB")
        target = self.label[idx]
        label = self.classes_name.index(target)
        if self.transform is not None:
            tensor_image = self.transform(image)
        return tensor_image, label


if __name__ == "__main__":
    trans = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(config.IMG_HEIGH, config.IMG_WIDTH))])

    print("------------ Creating Dataset ------------")
    
    full_dataset = ft_net_swin(config.IMG_PATH, trans)
    data_loader = torch.utils.data.DataLoader(
        full_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, drop_last=True 
    )