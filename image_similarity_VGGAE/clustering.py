from cv2 import transform
import model
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

    encoder = model.VGGEncoder()
    # Load the state dict of encoder
    encoder.load_state_dict(torch.load(config.ENCODER_MODEL_PATH, map_location=device))
    # Load the state dict of encoder
    encoder.eval()
    encoder.to(device)

    print("------------ Creating Dataset ------------")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(config.IMG_HEIGH, config.IMG_WIDTH))])
    
    print(config.TEST_DATA_PATH)
    full_dataset = data.IndoorDataset(config.TEST_DATA_PATH, transform)

    print("------------ Dataset Created ------------")

    print("------------ Creating DataLoader ------------")
   
    full_loader = torch.utils.data.DataLoader(
        full_dataset, batch_size=config.FULL_BATCH_SIZE
    )

    print("---- Creating Feature maps for the full dataset ---- ")

    layer_dict = {23 : (1, 256, 28, 28), 43 : (1, 512, 7, 7)}
    features = clustering_engine.create_feature(encoder, full_loader, layer_dict[23], device) # # EMBEDDING_SHAPE = (1, 517, 7, 7))
    
    
    
    
    # Convert embedding to numpy and save them
    numpy_features = features.cpu().detach().numpy()
    # print(numpy_embedding)
    num_images = numpy_features.shape[0]

    # Dump the embeddings for complete dataset, not just train
    flattened_features = numpy_features.reshape((num_images, -1))
    feature_path = os.path.join(config.FEATURE_PATH, "VGG16_23layers.npy")
    np.save(feature_path, flattened_features)


    print("Loads the feature")
    flattened_features = np.load(feature_path)
    test_image_path = "../test/korean-ReID/1/090401/IN_H00296_SN1_090401_25707.png"
    # distance_list, indices_list = clustering_engine.compute_similar_images(config.TEST_IMAGE_PATH, config.NUM_IMAGES, embedding, device)
    indices_list, distance_list = clustering_engine.compute_similar_images(test_image_path, flattened_features, device)
    clustering_engine.plot_similar_images(distance_list, indices_list, 1)
