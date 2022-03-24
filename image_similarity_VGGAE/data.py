__all__ = ["IndoorDataset", "TestDataset"]

import torch
from PIL import Image
import os
from torch.utils.data import Dataset
import glob
import config

# 한국인 재식별 이미지를 DataLoader로 사용하기 위한 작업
class IndoorDataset(Dataset):
    def __init__(self, main_dir, transform=None): # 데이터의 전처리를 해주는 부분
        self.main_dir = main_dir
        self.transform = transform        
        self.image_dir = glob.glob(main_dir + "/**/*.png", recursive=True) # 모든 png 파일의 이미지 경로 저장
        print("총 데이터 : ", len(self.image_dir))
        # 코드 test용  
        # tmp_dir = glob.glob(main_dir + "/**/*.png", recursive=True) # 모든 png 파일의 이미지 경로 저장
        # self.image_dir = tmp_dir[:64]

    def __len__(self): # 데이터셋의 길이, 즉 샘플의 수를 적어주는 부분
        return len(self.image_dir)

    def __getitem__(self, idx): # 데이터셋에서  특정 1개의 샘플을 가져오는 함수 
        img_loc = self.image_dir[idx]
        try:
            img = Image.open(img_loc)
            image = img.convert("RGB")
        except:
            print(self.image_dir[idx])
        finally:
            img.close() # 이미지 손상 될 수도 있기 때문에 코드 추가 

        if self.transform is not None:
            tensor_image = self.transform(image)

        return tensor_image, tensor_image


class TestDataset(Dataset):
    def __init__(self, main_dir, transform=None): # 데이터의 전처리를 해주는 부분
        self.main_dir = main_dir
        self.transform = transform        
        self.image_dir = glob.glob(main_dir + "/**/*.png", recursive=True) # 모든 png 파일의 이미지 경로 저장
        print("총 데이터 : ", len(self.image_dir))
        # 코드 test용  
        # tmp_dir = glob.glob(main_dir + "/**/*.png", recursive=True) # 모든 png 파일의 이미지 경로 저장
        # self.image_dir = tmp_dir[:64]

    def __len__(self): # 데이터셋의 길이, 즉 샘플의 수를 적어주는 부분
        return len(self.image_dir)

    def __getitem__(self, idx): # 데이터셋에서  특정 1개의 샘플을 가져오는 함수 
        img_loc = self.image_dir[idx]
        try:
            img = Image.open(img_loc)
            image = img.convert("RGB")
        except:
            print(self.image_dir[idx])
        finally:
            img.close() # 이미지 손상 될 수도 있기 때문에 코드 추가 

        if self.transform is not None:
            tensor_image = self.transform(image)

        return tensor_image, tensor_image

if __name__ == "__main__":
    pass