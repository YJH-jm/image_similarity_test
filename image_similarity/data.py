__all__ = ["IndoorDataset", "ClassificationDataset"], 

import torch
from PIL import Image
import os
from torch.utils.data import Dataset
import glob
import config
import torchvision.transforms as transforms


# 한국인 재식별 이미지를 DataLoader로 사용하기 위한 작업
class IndoorDataset(Dataset):
    def __init__(self, main_dir, transform=None): # 데이터의 전처리를 해주는 부분
        self.main_dir = main_dir
        self.transform = transform
        self.image_dir = glob.glob(main_dir + "/**/*.png", recursive=True)[:128] # 모든 png 파일의 이미지 경로 저장
        # print(type(self.image_dir))

    def __len__(self): # 데이터셋의 길이, 즉 샘플의 수를 적어주는 부분
        return len(self.image_dir)

    def __getitem__(self, idx): # 데이터셋에서  특정 1개의 샘플을 가져오는 함수 
        img_loc = self.image_dir[idx]
        image = Image.open(img_loc).convert("RGB")
        # print(img_loc)
        # image.show()
        label = self.image_dir[idx]
        # print(type(label))
        # print(label)
        if self.transform is not None:
            tensor_image = self.transform(image)

        return tensor_image, tensor_image


class ClassificationDataset(Dataset):
    def __init__(self, main_dir, transform=None): # 데이터의 전처리를 해주는 부분
        self.main_dir = main_dir
        self.transform = transform
        self.image_dir = glob.glob(main_dir + "/**/*.png", recursive=True) # 모든 png 파일의 이미지 경로 저장
        self.classes_name = set()
        self.label = []

        for i in self.image_dir:
            tmp = i.split('\\')[-1]
            # print(tmp)
            tmp = tmp.split('_')[1] + tmp.split('_')[2]
            # print(tmp)
            self.label.append(tmp)
            self.classes_name.add(tmp)
        self.classes_name = list(self.classes_name)
        self.classes_name.sort()
        print(self.classes_name)
        print(len(self.classes_name))
        

    def __len__(self): # 데이터셋의 길이, 즉 샘플의 수를 적어주는 부분
        return len(self.image_dir)

    def __getitem__(self, idx): # 데이터셋에서  특정 1개의 샘플을 가져오는 함수 
        img_loc = self.image_dir[idx]
        image = Image.open(img_loc).convert("RGB")
        # print(img_loc)
        # image.show()
        label = self.classes_name.index(self.label[idx])
        print(label)
        if self.transform is not None:
            tensor_image = self.transform(image)

        label = transforms.ToTensor()(label)
        print(label)
        return tensor_image, label

if __name__ == "__main__":
    trans = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(config.IMG_HEIGH, config.IMG_WIDTH))])

    print("------------ Creating Dataset ------------")
    
    full_dataset = ClassificationDataset(config.IMG_PATH, trans)
    data_loader = torch.utils.data.DataLoader(
        full_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, drop_last=True 
    )