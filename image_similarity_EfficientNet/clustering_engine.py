__all__ = ["load_image_tensor", "create_feature"]

from pyexpat import features
from efficientnet_pytorch import EfficientNet 
import torch
import torchvision.transforms as transforms
import config 
from sklearn.neighbors import NearestNeighbors 

from PIL import Image
import matplotlib.pyplot as plt
import glob

def load_image_tensor(image_path, device):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(config.IMG_HEIGH, config.IMG_WIDTH)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    image_tensor = transform((Image.open(image_path)))
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

def create_feature(model, full_loader, feature_dim, device): # save image feature representations of all images in the dataset
    
    # Load the state dict of encoder
    
    # encoder.load_state_dict(torch.load(config.ENCODER_MODEL_PATH, map_location=device))
    model.eval()

    features = torch.randn(feature_dim)
    # print("check : ", embedding.shape)
    
    with torch.no_grad():
        for _, (train_img, target) in enumerate(full_loader):
            train_img = train_img.to(device)
            target = target.to(device)
            outputs = model.extract_features(train_img)           
            features = torch.cat((features, outputs), 0)

    features = features[1:, :, :, :]

    return features


def get_label():
    label = []
    image_dir = glob.glob(config.TEST_DATA_PATH + "/**/*.png", recursive=True) # 모든 png 파일의 이미지 경로 저장
    for i in image_dir:
        i = i.split('\\')[2]
        label.append(i)
    return label, image_dir

def compute_similar_images(image_path, features, device):
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=config.NUM_CLASSES) #
    model.load_state_dict(torch.load(config.EFFICIENTNET_MODEL_PATH , map_location=device))

    model.eval()
    model.to(device)

    image_tensor = load_image_tensor(image_path, device)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        image_feature = model.extract_features(image_tensor).cpu().detach().numpy()

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
    # get_label()

    model = EfficientNet()
    
    
    img_random_VGG = torch.randn(1, 3, config.IMG_HEIGH, config.IMG_WIDTH)
    img_random_VGG2 = torch.randn(1, 3, config.IMG_HEIGH, config.IMG_WIDTH)

