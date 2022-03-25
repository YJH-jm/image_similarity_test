# Training script for VGG Auto-Encoder.

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm

import config
import data
import model
import train_engine
import utils

if __name__ == "__main__":
    print(torch.cuda.is_available())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    print("Setting Seed for the run, seed = {}".format(config.SEED))

    utils.seed_everything(config.SEED)

    transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(config.IMG_HEIGH, config.IMG_WIDTH))])

    print("------------ Creating Dataset ------------")
    
    full_dataset = data.IndoorDataset(config.IMG_PATH, transforms)
    # print(len(full_dataset)) # 1744

    train_size = int(config.TRAIN_RATIO * len(full_dataset)) 
    val_size = len(full_dataset) - train_size 

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size] 
    )

    print("------------ Dataset Created ------------")
    print("------------ Creating DataLoader ------------")
   
   
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, drop_last=True
    ) 

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.TEST_BATCH_SIZE
    )
    full_loader = torch.utils.data.DataLoader(
        full_dataset, batch_size=config.FULL_BATCH_SIZE
    )

    print("------------ Dataloader Cretead ------------")

    if torch.cuda.is_available():
        print("GPU Availaible moving models to GPU")
    else:
        print("Moving models to CPU")
    
    

    encoder = model.VGGEncoder()
    decoder = model.VGGDecoder(encoder.encoder)


    encoder.to(device)
    decoder.to(device)


    criterion = nn.MSELoss()
    autoencoder_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(autoencoder_params, lr=config.LEARNING_RATE)

    # early_stopper = utils.EarlyStopping(patience=5, verbose=True, path=)
    min_loss = 9999

    print("------------ Training started ------------")

    for epoch in tqdm(range(config.EPOCHS)):
        train_loss = train_engine.train_step(
            encoder, decoder, train_loader, criterion, optimizer, device=device
        )
        print(f"Epochs = {epoch + 1}, Training Loss : {train_loss}")
        val_loss = train_engine.val_step(
            encoder, decoder, val_loader, criterion, device=device
        )

        # Simple Best Model saving
        if val_loss < min_loss:
            print("Validation Loss decreased, saving new best model")
            torch.save(encoder.state_dict(), config.ENCODER_MODEL_PATH)
            # torch.nn.Module에서 모델로 학습할 때 각 layer마다 텐서로 매핑되는 매개변수(예를 들어 가중치, 편향과 같은)를 python dictionary 타입으로 저장한 객체
            # 학습 가능한 매개변수를 가지는 layer만이 모델의 state_dict에 항목을 가짐
            torch.save(decoder.state_dict(), config.DECODER_MODEL_PATH)
            min_loss = val_loss # 이 코드 필요한 것 같g은데? 아닌가..?
            print("min_loss : ", min_loss )

        print(f"Epochs = [{epoch+1}/{config.EPOCHS}], Validation Loss : {val_loss}")

    print("Training Done")
