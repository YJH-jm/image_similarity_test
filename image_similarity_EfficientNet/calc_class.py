__all__ = ["IndoorDataset"]

from cv2 import transform
import torch
from PIL import Image
import os
from torch.utils.data import Dataset
import glob
import config
import torchvision.transforms as transforms
# 한국인 재식별 이미지를 DataLoader로 사용하기 위한 작업


if __name__ == "__main__":
    num_classes = glob.glob(config.IMG_PATH + "/**/SN*", recursive=True)
    classes = set()
    num1 = 0
    num2 = 0
    num3 = 0
    num4 = 0
    others = 0
    for i in range(len(num_classes)):
        a = num_classes[i]
        a_list = a.split('\\')
        # print(a_list[-1])
        classes.add(a_list[-1])
        if a_list[-1] == "SN1":
            num1 += 1
        elif a_list[-1] == "SN2":
            num2 += 1
        elif a_list[-1] == "SN3":
            num3 += 1
        elif a_list[-1] == "SN4":
            num4 += 1
        else:
            others += 1
            print(a_list)
    print(num1, num2, num3, num4, others)
    print(classes)
# 1603 개의 class 