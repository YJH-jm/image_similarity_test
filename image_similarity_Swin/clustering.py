from efficientnet_pytorch import EfficientNet 
import torch
import torchvision.transforms as transforms
import config
import clustering_engine
import numpy as np
import data
import os

if __name__ == "__main__":

    # Loads the model

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load the state dict of encoder
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=config.NUM_CLASSES) #
    model.load_state_dict(torch.load(config.EFFICIENTNET_MODEL_PATH , map_location=device))
    

    print("------------ Creating Dataset ------------")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(config.IMG_HEIGH, config.IMG_WIDTH)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    full_dataset = data.TestDataset(config.TEST_DATA_PATH, transform)
    
    print("------------ Dataset Created ------------")


    print("------------ Creating DataLoader ------------")
   
    full_loader = torch.utils.data.DataLoader(
        full_dataset, batch_size=config.FULL_BATCH_SIZE
    )

    print("------------ Dataloader Cretead ------------")


    print("---- Creating Feature maps for the full dataset ---- ")

    features = clustering_engine.create_feature(model, full_loader, config.FEATURE_SHAPE, device) # # EMBEDDING_SHAPE = (1, 517, 7, 7))
    
    
    
    
    # Convert features to numpy and save them
    numpy_features = features.cpu().detach().numpy()
    num_images = numpy_features.shape[0]

    # Dump the embeddings for complete dataset, not just train
    flattened_features = numpy_features.reshape((num_images, -1))
    np.save(config.FEATURE_PATH, flattened_features)


    # print("Loads the feature")
    # flattened_features = np.load(config.FEATURE_PATH)
    # test_image_path = "../test/korean-ReID/1/090401/IN_H00296_SN1_090401_25707.png"
    # test_image_path = "../test/korean-ReID/2/090407/IN_H00296_SN1_090407_21582.png"
    # test_image_path = "../test/korean-ReID/3/102504/IN_H00803_SN2_102504_14915.png"
    # test_image_path = "../test/korean-ReID/4/090404/IN_H00276_SN1_090404_9000.png"
    # test_image_path = "../test/korean-ReID/5/090405/IN_H00276_SN2_090405_20271.png"
    # test_image_path = "../test/korean-ReID/6/101905/IN_H00702_SN1_101905_29150.png"
    # test_image_path = "../test/korean-ReID/7/101904/IN_H00702_SN3_101904_14390.png"
    # test_image_path = "../test/korean-ReID/8/101903/IN_H00711_SN1_101903_19150.png"
    # test_image_path = "../test/korean-ReID/9/101903/IN_H00724_SN1_101903_23847.png"
    # test_image_path = "../test/korean-ReID/10/101909/IN_H00724_SN1_101909_29902.png"
    
    # 모든 이미지를 KNeighbors 로 clustering 하여 label과 distance 를 얻는 함수 
    all_labels, all_img_dir = clustering_engine.get_label()
    total_label_list, total_distance_list = clustering_engine.compute_all_similar_images(model, device, all_labels, all_img_dir)
    # K-means 알고리즘으로 grouping 한 label 획득
    group_label = clustering_engine.clustering()

    # 글로벌 ID 통합
    clustering_engine.test(total_label_list, total_distance_list, all_labels, all_img_dir, group_label)
    

    # clustering_engine.compute_all_similar_images(full_dataset, device)

    # # distance_list, indices_list = clustering_engine.compute_similar_images(config.TEST_IMAGE_PATH, config.NUM_IMAGES, embedding, device)
    # indices_list, distance_list = clustering_engine.compute_similar_images(test_image_path, flattened_features, device)
    # clustering_engine.plot_similar_images(distance_list, indices_list, 1)