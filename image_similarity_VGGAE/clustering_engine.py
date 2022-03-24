__all__ = ["load_image_tensor", "create_feature"]

from pyexpat import features
import model
import torch
import torchvision.transforms as transforms
import config 
from sklearn.neighbors import NearestNeighbors 

from PIL import Image
import matplotlib.pyplot as plt
import glob

def load_image_tensor(image_path, device):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(config.IMG_HEIGH, config.IMG_WIDTH))])
    image_tensor = transform((Image.open(image_path)))
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

def create_feature(encoder, full_loader, feature_dim, device): # save image feature representations of all images in the dataset
    print(feature_dim)
    print(type(feature_dim))

    encoder = model.VGGEncoder()
    decoder = model.VGGDecoder(encoder.encoder)

    # Load the state dict of encoder
    encoder.load_state_dict(torch.load(config.ENCODER_MODEL_PATH, map_location=device))
    encoder.eval()

    features = torch.randn(feature_dim)
    # print("check : ", embedding.shape)

    with torch.no_grad():
        for _, (train_img, _) in enumerate(full_loader):
            
            train_img = train_img.to(device)
            enc_output= encoder.get_feature(train_img)
            # print("embadding enc_output shape : \n", enc_output.shape) #  torch.Size([32, 256, 16, 16])
            features= torch.cat((features, enc_output), 0)
            # print("embadding embadding shape : \n", embedding.shape)
            # print("-"*30)

    features = features[1:, :, :, :]
    print("최종 embedding shape : ", features.shape)

    return features


def get_label():
    label = []
    image_dir = glob.glob(config.TEST_DATA_PATH + "/**/*.png", recursive=True) # 모든 png 파일의 이미지 경로 저장
    for i in image_dir:
        i = i.split('\\')[2]
        label.append(i)
    return label, image_dir

def compute_similar_images(image_path, features, device):
    encoder = model.VGGEncoder()
    encoder.load_state_dict(torch.load(config.ENCODER_MODEL_PATH, map_location=device))

    encoder.eval()
    encoder.to(device)

    image_tensor = load_image_tensor(image_path, device)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        image_feature = encoder.get_feature(image_tensor).cpu().detach().numpy()

    flattened_feature = image_feature.reshape((image_feature.shape[0], -1))

    # KNN
    knn = NearestNeighbors(n_neighbors=config.NUM_NEIGHBORS, algorithm="brute", metric="cosine")
    knn.fit(features)

    distance, indices = knn.kneighbors(flattened_feature)
    indices_list = indices.tolist()
    distance_list = distance.tolist()

    return indices_list, distance_list


def plot_similar_images(distance_list, indices_list, choice): # plot_similar_images(distance_list, indices_list, 1)
    """
    Plots images that are similar to indices obtained from computing simliar images.
    Args:
    indices_list : List of List of indexes. E.g. [[1, 2, 3]]
    """

    indices = indices_list[0]
    distance = distance_list[0]
    
    label, image_dir = get_label()

    plt.figure(figsize= (14, 42))
    
    for index in range(len(distance)):
        img_idx = int(indices[index])
        print(img_idx)
        img_path = image_dir[img_idx]
        print(img_path)
        img = Image.open(img_path).convert("RGB")
        plt.subplot(2, 6, index+1)
        title = str(label[img_idx])+'\n' + str(distance[index])
        plt.title(title)
        plt.imshow(img)
    # plt.savefig('save.jpg')
    plt.show()

if __name__ == "__main__":
    get_label()

    encoder = model.VGGEncoder()
    
    img_random_VGG = torch.randn(1, 3, config.IMG_HEIGH, config.IMG_WIDTH)
    img_random_VGG2 = torch.randn(1, 3, config.IMG_HEIGH, config.IMG_WIDTH)

    for module_encoder in encoder:
        print(module_encoder)