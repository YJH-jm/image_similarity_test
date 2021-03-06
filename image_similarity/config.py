IMG_PATH = "../input/korean-ReID/"
IMG_HEIGH = 224  # The images are already resized here
IMG_WIDTH = 224  # The images are already resized here

SEED = 42
TRAIN_RATIO = 0.75
VAL_RATIO = 1 - TRAIN_RATIO
SHUFFLE_BUFFER_SIZE = 100

LEARNING_RATE = 1e-3
EPOCHS = 30
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
FULL_BATCH_SIZE = 32

###### Train and Test time #########

# TEST_DATA_PATH = "../data/images/"
TEST_DATA_PATH = "../input/korean-ReID/" # 일단 같은 이미지 확인
AUTOENCODER_MODEL_PATH = "VGG_autoencoder.pt"
ENCODER_MODEL_PATH = "../data/models/VGG_encoder.pt"
DECODER_MODEL_PATH = "../data/models/VGG_decoder.pt"
EMBEDDING_PATH = "../data/models/data_VGG_embedding_f.npy"
EMBEDDING_SHAPE = (1, 512, 7, 7)

# TEST_RATIO = 0.2

###### Test time #########
# NUM_IMAGES = 10
NUM_IMAGES = 5

# TEST_IMAGE_PATH = "../data/images/test.jpg"
# TEST_IMAGE_PATH = "../data/images/4625.jpg"
