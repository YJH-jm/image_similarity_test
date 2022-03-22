__all__ = ["train_step", "val_step", "create_embedding"]

import torch
import torch.nn as nn
import model
import config 


def train_step(encoder, decoder, train_loader, criterion, optimizer, device):
    # Set networks to train mode
    encoder.train()
    decoder.train()

    # print(device)

    for batch_idx, (train_img, target_img) in enumerate(train_loader):
        print(f"Steps : [{batch_idx}]")
        train_img = train_img.to(device)
        target_img = target_img.to(device)

        optimizer.zero_grad()

        enc_output, pooling_indices = encoder(train_img)
        dec_output = decoder(enc_output, pooling_indices)

        loss = criterion(dec_output, target_img)
        loss.backward()

        optimizer.step()

    return loss.item()


def val_step(encoder, decoder, val_loader, criterion, device):

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for batch_idx, (train_img, target_img) in enumerate(val_loader):
            print(f"Steps : [{batch_idx}]")
            train_img = train_img.to(device)
            target_img = target_img.to(device)

            enc_output, pooling_indices = encoder(train_img)
            dec_output = decoder(enc_output, pooling_indices)

            loss = criterion(dec_output, target_img)

    return loss.item()

def create_embedding(encoder, full_loader, embedding_dim, device): # save image feature representations of all images in the dataset
    encoder = model.VGGEncoder()
    decoder = model.VGGDecoder(encoder.encoder)

    # Load the state dict of encoder
    encoder.load_state_dict(torch.load(config.ENCODER_MODEL_PATH, map_location=device))
    encoder.eval()
    embedding = torch.randn(embedding_dim)
    # print("check : ", embedding.shape)

    with torch.no_grad():
        for batch_idx, (train_img, target_img) in enumerate(full_loader):
            print(train_img[0])
            train_img = train_img.to(device)
            enc_output, _ = encoder(train_img)
            # print("embadding enc_output shape : \n", enc_output.shape) #  torch.Size([32, 256, 16, 16])
            embedding = torch.cat((embedding, enc_output), 0)
            # print("embadding embadding shape : \n", embedding.shape)
            # print("-"*30)

    print("최종 embedding shape : ", embedding.shape)

    return embedding
