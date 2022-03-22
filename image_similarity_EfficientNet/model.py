__all__ = ["VGGEncoder", "VGGDecoder"]

import torch
import torch.nn as nn
from torchvision import models
import config 
from efficientnet_pytorch import EfficientNet 

class Efficient():
    def __init__(self):
        # self.model = EfficientNet.from_name('efficientnet-b0') # 모델만 로드
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=config.NUM_CLASSES) # 모델 + 학습된 weight

    def forward(self, x):
        output = self.model(x)
        return output

        # print(self.model.shape)
        # return x_current, pool_indices

    def parameters(self):
        return self.model.parameters()

    def to(self, device):
        return self.model.to(device)


################################################################################################################################################



if __name__ == "__main__":
    img_random = torch.randn(1, 3, config.IMG_HEIGH, config.IMG_WIDTH)
    img_random_VGG2 = torch.randn(1, 3, config.IMG_HEIGH, config.IMG_WIDTH)

    model = Efficient()
    feature_map = model.forward(img_random)
    print(feature_map.shape)