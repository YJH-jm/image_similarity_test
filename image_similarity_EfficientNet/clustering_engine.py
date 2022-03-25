__all__ = ["load_image_tensor", "create_feature"]

from pyexpat import features
from efficientnet_pytorch import EfficientNet 
import torch
import torchvision.transforms as transforms
import config 
from sklearn.neighbors import NearestNeighbors 
from sklearn.cluster import KMeans, DBSCAN
from PIL import Image
import matplotlib.pyplot as plt
import glob
import numpy as np

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


def get_label(img_path=config.TEST_DATA_PATH):
    label = []
    image_dir = glob.glob(config.TEST_DATA_PATH + "/**/*.png", recursive=True) # 모든 png 파일의 이미지 경로 저장
    # ReID TEST 데이터의 경우 label 이런 식으로 구성
    for i in image_dir:
        i = i.split('\\')[2]
        label.append(i)
    
    return label, image_dir


# 하나의 이미지에 대해 clustering 하기 위한 함수
def compute_similar_images(model, image_path, features, device):


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


def compute_all_similar_images(model, device, all_labels, all_img_dir):
    
    # model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=config.NUM_CLASSES) #
    # model.load_state_dict(torch.load(config.EFFICIENTNET_MODEL_PATH , map_location=device))
    model.eval()
    model.to(device)
    
    print("Loads the feature")
    all_features = np.load(config.FEATURE_PATH)

    total_label_list = []
    total_distance_list = []
    
    for idx in range(len(all_img_dir)):
        indices_list, distance_list = compute_similar_images(model, all_img_dir[idx], all_features, device)
        distances = distance_list[0]
        indices = indices_list[0]
        labels = []
        for i in range(len(indices)):
            labels.append(all_labels[indices[i]])
        print(labels)
        total_label_list.append(labels)
        total_distance_list.append(distances)
    
    # print(len(total_distance_list[0])) # 12
    # print(len(total_label_list)) # 60
    
    return total_label_list, total_distance_list

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

def test(total_label_list, total_distance_list, all_labels, all_img_dir):
    # print(len(total_distance_list)) # 60
    # print(len(total_distance_list[0])) # 12
    # print(len(total_label_list)) # 60
    # print(len(total_label_list[0])) # 12

    cur_label = all_labels[0]
    for num in range(len(total_label_list)): # 하나의 이미지
        if cur_label != all_labels[num]:
            cur_label = all_labels[num]
        for idx in range(len(total_label_list[0])): # 12개의 결과
            pass

def clustering():
    print("Loads the feature")
    all_features = np.load(config.FEATURE_PATH)
    km = KMeans(n_clusters=10, random_state = 0)
    km.fit(all_features)
    print(km.labels_)
    print(km)

    ds = DBSCAN(eps=100)
    ds.fit(all_features)
    print(ds.labels_)    
if __name__ == "__main__":
    # get_label()

    clustering()
    # model = EfficientNet()
    
    
    # img_random_VGG = torch.randn(1, 3, config.IMG_HEIGH, config.IMG_WIDTH)
    # img_random_VGG2 = torch.randn(1, 3, config.IMG_HEIGH, config.IMG_WIDTH)

