IMG_PATH = "../input/korean-ReID/"
IMG_HEIGH = 224  
IMG_WIDTH = 224 

SEED = 42
TRAIN_RATIO = 0.75
VAL_RATIO = 1 - TRAIN_RATIO
SHUFFLE_BUFFER_SIZE = 100

LEARNING_RATE = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EPOCHS = 30
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
FULL_BATCH_SIZE = 32

###### Train and Test time #########

TEST_DATA_PATH = "../test"
SWIN_MODEL_PATH = "../data/models/Swin.pt"
FEATURE_PATH = "../data/models/Swin_features.npy"
# FEATURE_PATH = "../data/models/EFF_features2.npy"

FEATURE_SHAPE = (1, 1280, 7, 7)
NUM_CLASSES=1603


###### Test time #########
NUM_IMAGES = 12
# TEST_IMAGE_PATH = "../data/images/test.jpg"
# TEST_IMAGE_PATH = "../data/images/4625.jpg"


#KNN
NUM_NEIGHBORS = 12