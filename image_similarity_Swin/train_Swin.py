# Training script for VGG Auto-Encoder.

import torch
import train_engine
import torchvision.transforms as transforms
from model import ft_net_swin
import data
import config
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import utils
import os
import model
from torch.utils.data.sampler import SubsetRandomSampler
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("Setting Seed for the run, seed = {}".format(config.SEED))

    utils.seed_everything(config.SEED)

    transforms_train = transforms.Compose([
        #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((config.IMG_HEIGH, config.IMG_WIDTH), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((config.IMG_HEIGH, config.IMG_WIDTH)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    transforms_val = transforms.Compose([
        transforms.Resize(size=(config.IMG_HEIGH, config.IMG_WIDTH),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    print("------------ Creating Dataset ------------")
    
    full_dataset = data.TrainDataset(config.IMG_PATH, transforms_val)
    train_dataset = data.TrainDataset(config.IMG_PATH, transforms_train)
    val_dataset = data.TrainDataset(config.IMG_PATH, transforms_val)
    # print(len(full_dataset)) 

    num_data = len(full_dataset)
    print(num_data)
    indices = list(range(num_data))
    
    train_size = int(config.TRAIN_RATIO * len(full_dataset)) 
    val_size = len(full_dataset) - train_size 
    np.random.shuffle(indices)
    train_idx, val_idx = indices[:train_size], indices[train_size:]
    # print(train_idx)
    # print(val_idx)
    train_sampler =SubsetRandomSampler(indices=train_idx)
    val_sampler = SubsetRandomSampler(indices=val_idx)


    print(train_dataset)
    print(val_dataset)
    print("------------ Dataset Created ------------")


    print("------------ Creating DataLoader ------------")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, sampler = train_sampler, shuffle=False, drop_last=True,  
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.TEST_BATCH_SIZE, sampler = val_sampler
    )

    full_loader = torch.utils.data.DataLoader(
        full_dataset, batch_size=config.FULL_BATCH_SIZE
    )

    print("------------ Dataloader Cretead ------------")

    if torch.cuda.is_available():
        print("GPU Availaible moving models to GPU")
    else:
        print("Moving models to CPU")
    
    

    swin = model.ft_net_swin(config.NUM_CLASSES)
    swin.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(swin.parameters(), momentum=config.MOMENTUM, nesterov=True,
                              lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    # optimizer = optim.AdamW(swin.parameters(), eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
    #                             lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    # optimizer = optim.SGD(swin.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    lr_ = lambda epoch : 0.98739
    exp_lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_)


    min_loss = 9999

    print("------------ Training started ------------")

    for epoch in tqdm(range(config.EPOCHS)):
        train_loss = train_engine.train_step(
            swin, train_loader, criterion, optimizer, device=device
        )
        print(f"Epochs = {epoch + 1}, Training Loss : {train_loss}")
        val_loss, val_accuracy = train_engine.val_step(
            swin, val_loader, criterion, device=device
        )

        # Simple Best Model saving
        if val_loss < min_loss:
            print("Validation Loss decreased, saving new best model")
            torch.save(swin.state_dict(), config.SWIN_MODEL_PATH)
            # torch.nn.Module에서 모델로 학습할 때 각 layer마다 텐서로 매핑되는 매개변수(예를 들어 가중치, 편향과 같은)를 python dictionary 타입으로 저장한 객체
            # 학습 가능한 매개변수를 가지는 layer만이 모델의 state_dict에 항목을 가짐
            min_loss = val_loss # 이 코드 필요한 것 같g은데? 아닌가..?
            print("min_loss : ", min_loss )

        print(f"Epochs = [{epoch+1}/{config.EPOCHS}], Validation Loss : {val_loss}, Accuracy : {val_accuracy}")
        print("min loss : ", min_loss)
        print("-" * 15)

    print("Training Done")
