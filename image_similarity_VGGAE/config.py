IMG_PATH = "../input/korean-ReID/"
IMG_HEIGH = 224  # VGG INPUT SIZE
IMG_WIDTH = 224  # VGG INPUT SIZE

SEED = 42
TRAIN_RATIO = 0.75
VAL_RATIO = 1 - TRAIN_RATIO
SHUFFLE_BUFFER_SIZE = 100

LEARNING_RATE = 1e-3
EPOCHS = 30
# TRAIN_BATCH_SIZE = 32 # CPU OOM error
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16
FULL_BATCH_SIZE = 16


TEST_DATA_PATH = "../test"
# TEST_DATA_PATH = "../input/korean-ReID/" # 일단 같은 이미지 확인
# AUTOENCODER_MODEL_PATH = "VGG_autoencoder.pt"
ENCODER_MODEL_PATH = "../data/models/VGG_encoder.pt"
DECODER_MODEL_PATH = "../data/models/VGG_decoder.pt"
FEATURE_PATH = "../data/models"
FEATURE_SHAPE = (1, 512, 7, 7)

# TEST_RATIO = 0.2

###### Test time #########
# NUM_IMAGES = 10

NUM_IMAGES = 12

# TEST_IMAGE_PATH = "../data/images/test.jpg"
# TEST_IMAGE_PATH = "../data/images/4625.jpg"

#KNN
NUM_NEIGHBORS = 12