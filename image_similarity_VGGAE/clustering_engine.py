__all__ = ["load_image_tensor", "create_feature"]

import model
import torch
import torchvision.transforms as transforms
import config 
from sklearn.neighbors import NearestNeighbors 

from PIL import Image


def load_image_tensor(image_path, device):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(config.IMG_HEIGH, config.IMG_WIDTH))])
    image_tensor = transform((Image.open(image_path)))
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

def create_feature(encoder, full_loader, feature_dim, device): # save image feature representations of all images in the dataset
    encoder = model.VGGEncoder()
    decoder = model.VGGDecoder(encoder.encoder)

    # Load the state dict of encoder
    encoder.load_state_dict(torch.load(config.ENCODER_MODEL_PATH, map_location=device))
    encoder.eval()
    embedding = torch.randn(feature_dim)
    # print("check : ", embedding.shape)

    with torch.no_grad():
        for batch_idx, (train_img, target_img) in enumerate(full_loader):
            train_img = train_img.to(device)
            enc_output, _ = encoder(train_img)
            # print("embadding enc_output shape : \n", enc_output.shape) #  torch.Size([32, 256, 16, 16])
            embedding = torch.cat((embedding, enc_output), 0)
            # print("embadding embadding shape : \n", embedding.shape)
            # print("-"*30)

    print("최종 embedding shape : ", embedding.shape)

    return embedding

def compute_similar_images(image_path, features, device):
    encoder = model.VGGEncoder()
    encoder.load_state_dict(torch.load(config.ENCODER_MODEL_PATH))

    encoder.eval()
    encoder.to(device)

    image_tensor = load_image_tensor(image_path, device)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        image_feature = encoder(image_tensor)[0].cpu().detach().numoy()

    flattened_feature = image_feature.reshape((image_feature.shape[0], -1))

    # KNN
    knn = NearestNeighbors(n_neighbors=config.NUM_NEIGHBORS, algorithm="brute", metric="cosine")
    knn.fit(features)

    distance, indices = knn.kneighbors(flattened_feature)
    indices_list = indices.tolist()
    distance_list = distance.tolist()

    return indices_list, distance_list
