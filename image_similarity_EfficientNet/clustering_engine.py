__all__ = ["load_image_tensor", "create_feature", "get_label", "compute_similar_images", "compute_all_similar_images", "plot_similar_images", "test", "clustering"]

import shutil
from efficientnet_pytorch import EfficientNet 
import torch
import torchvision.transforms as transforms
import config 
from sklearn.neighbors import NearestNeighbors 
from sklearn.cluster import KMeans, DBSCAN
from PIL import Image
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

def load_image_tensor(image_path, device):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(config.IMG_HEIGH, config.IMG_WIDTH)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    image_tensor = transform((Image.open(image_path)))
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

def create_feature(model, full_loader, feature_dim, device): # save image feature representations of all images in the dataset
    
    
    model.eval()
    features = torch.randn(feature_dim)
    
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
    # print(total_label_list)
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
    plt.show()

def test(total_label_list, total_distance_list, all_labels, all_img_dir, group_label):
    label_result = dict()
    for idx in range(len(all_labels)):
        cur_label = all_labels[idx]
        label_dict = dict()
        for i in range(6):
            label = total_label_list[idx][i]
            if label == cur_label:
                continue

            if label in label_dict:
                label_dict[label] += 1
            else:
                label_dict[label] = 1

        if len(label_dict) == 1:
            for _, value in label_dict.items():
                if value >= 3:
                    if str(cur_label) in label_result:
                        label_result[cur_label].append(label_dict)
                    else:
                        label_result[cur_label] = list()
                        label_result[cur_label].append(label_dict)
    print(label_result)
    # {'1': [{'2': 3}, {'2': 3}, {'2': 3}, {'2': 3}, {'2': 3}], '10': [{'9': 3}, {'9': 4}, {'9': 3}, {'9': 3}], '2': [{'1': 4}, {'1': 3}, {'1': 3}, {'1': 3}, {'1': 3}, {'1': 3}], '6': [{'7': 3}], '9': [{'10': 3}, {'10': 3}, {'10': 4}, {'10': 3}, {'10': 3}, {'10': 3}]}
    
    check = set()
    for key, value in label_result.items():
        value_list = list()
        label_list = list()
        if len(value) < 3:
            continue

        for lst in value: 
            flag = False
            print(lst) # {'2' : 3}
            for k , v in lst.items():
                if k not in label_list:
                    label_list.append(k)
                    value_list.append(1)
                else:
                    idx = label_list.index(k)
                    value_list[idx] += 1

                    if value_list[idx] >= 3:
                        tuples = (int(key), int(k))
                        tuples = sorted(tuples)
                        tuples = tuple(tuples)
                        check.add(tuples)
                        flag = True
                if flag:
                    break

    print(check) # {(1, 2), (9, 10)}
    
    for id1, id2 in check:
        cnt = 0
        flag = False
        for idx in range(len(all_labels)):
            if all_labels[idx] == id1 or all_labels[idx] == id2:
                if cnt == 0:
                    group_num = group_label[idx]
                    cnt += 1
                else:
                    if group_num != group_label[idx]:
                        flag = True
                        break
                
                if cnt == 12:
                    break

        
        if not flag:
            print("폴더 이동")
            source = os.path.join(config.TEST_DATA_PATH, "korean-ReID", str(id2))
            print(source)
            destination = os.path.join(config.TEST_DATA_PATH, "korean-ReID", str(id1))
            print(destination)

            get_files = os.listdir(source)
            # print("-"*10)
            for g in get_files:
                # print(g)
                path = os.path.join(source, g)
                # print(path)
                shutil.move(path, destination)
                
def clustering():
    print("Loads the feature")
    all_features = np.load(config.FEATURE_PATH)
    km = KMeans(n_clusters=6, random_state = 0)
    km.fit(all_features)
    print(km.labels_)
    
    return km.labels_
 
if __name__ == "__main__":
    pass