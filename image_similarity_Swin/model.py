__all__ = ["swin", "VGGDecoder"]

import torch
import torch.nn as nn
from torchvision import models
import config
import timm


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)

class ClassBlock(nn.Module): # ClassBlock(1024, class_num, droprate, linear=linear_num, return_f = circle)
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, linear=1024, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear>0:
            add_block += [nn.Linear(input_dim, linear)]
        else:
            linear = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(linear)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(linear, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x,f]
        else:
            x = self.classifier(x)
            return x
      
class ft_net_swin(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, circle=False, linear_num=1024):
        super(ft_net_swin, self).__init__()
        model_ft = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        # avg pooling to global pooling
        #model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.head = nn.Sequential() # save memory
        self.model = model_ft
        self.circle = circle
        self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f = circle)
        

    def forward(self, x):
        x = self.model.forward_features(x)
        # print("feature_map shape : ", x.shape) # torch.Size([1, 1024])
        x = self.classifier(x)
        return x

    def get_feature(self, x):
        


################################################################################################################################################



if __name__ == "__main__":
    img_random_1 = torch.randn(1, 3, config.IMG_HEIGH, config.IMG_WIDTH)
    img_random_2 = torch.randn(1, 3, config.IMG_HEIGH, config.IMG_WIDTH)

    model = ft_net_swin(config.NUM_CLASSES)
    features = model.get_
    model(img_random_1)