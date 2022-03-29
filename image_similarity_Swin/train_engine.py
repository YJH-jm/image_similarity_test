__all__ = ["train_step", "val_step", "create_embedding"]

import torch
import torch.nn as nn
import config



def train_step(model, train_loader, criterion, optimizer, device):
    # Set networks to train mode
    model.train()

    # print(device)

    for batch_idx, (train_img, target) in enumerate(train_loader):
        # print(f"Steps : [{batch_idx}]")
        train_img = train_img.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        outputs = model(train_img)
        loss = criterion(outputs, target)
        loss.backward()

        optimizer.step()

    return loss.item()

def val_step(model, val_loader, criterion, device):

    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for batch_idx, (train_img, target) in enumerate(val_loader):
            # print(f"Steps : [{batch_idx}]")
            train_img = train_img.to(device)
            target = target.to(device)

            outputs = model(train_img)
            loss = criterion(outputs, target)
            
            _, predicted = torch.max(outputs.data, 1)
            # print("outputs.data shape : ", outputs.data.shape)
            # print("predicted shape", predicted.shape)
            # print("predicted : ", predicted)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            accuracy  = 100 * correct / total
    return loss.item(), accuracy


def create_embedding(encoder, full_loader, embedding_dim, device): # save image feature representations of all images in the dataset
    model = EfficientNet()

    # Load the state dict of encoder
    model.load_state_dict(torch.load(config.EFFICIENTNET_MODEL_PATH, map_location=device))
    model.eval()
    embedding = torch.randn(embedding_dim)
    # print("check : ", embedding.shape)

    with torch.no_grad():
        for batch_idx, (train_img, target) in enumerate(full_loader):
            train_img = train_img.to(device)
            feature = model.extract_features(train_img)
            # print("embadding enc_output shape : \n", enc_output.shape) #  torch.Size([32, 256, 16, 16])
            embedding = torch.cat((embedding, feature), 0)
            # print("embadding embadding shape : \n", embedding.shape)
            # print("-"*30)

    print("최종 embedding shape : ", embedding.shape)

    return embedding
