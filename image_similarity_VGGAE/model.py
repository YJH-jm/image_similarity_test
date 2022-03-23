__all__ = ["VGGEncoder", "VGGDecoder"]

import torch
import torch.nn as nn
from torchvision import models
import config 

class VGGEncoder(nn.Module):
    '''Encoder of image based on the architecture of VGG-16 with batch normalization.
    Args:
        pretrained_params (bool, optional): If the network should be populated with pre-trained VGG parameters.
            Defaults to True.
    '''

    channels_in = 3
    channels_latent = 512

    def __init__(self, pretrained_params=True):
        super(VGGEncoder, self).__init__()

        vgg = models.vgg16_bn(pretrained=pretrained_params)
        del vgg.classifier
        del vgg.avgpool

        self.encoder = self._encodify_(vgg)

    def _encodify_(self, encoder):
        '''
        Args:
            encoder : VGG model
        Returns:
            modules : VGG16의 encoder 목록
        '''
        modules = nn.ModuleList()
        for module in encoder.features:
            if isinstance(module, nn.MaxPool2d):
                module_add = nn.MaxPool2d(kernel_size=module.kernel_size,
                                          stride=module.stride,
                                          padding=module.padding,
                                          return_indices=True)
                modules.append(module_add)
            else:
                modules.append(module)
        return modules
    
    
    def forward(self, x):
        '''
        Args:
            x (Tensor): 이미지 tensor
        Returns:
            x_code (Tensor): code tensor
            pool_indices (list): Pool indices tensors in order of the pooling modules
        '''
        pool_indices = []
        x_current = x
        for module_encode in self.encoder:
            output = module_encode(x_current)
            # If the module is pooling, there are two outputs, the second the pool indices
            if isinstance(output, tuple) and len(output) == 2:
                x_current = output[0]
                pool_indices.append(output[1])
            else:
                x_current = output

        return x_current, pool_indices

class VGGDecoder(nn.Module):
    '''Decoder of code based on the architecture of VGG-16 with batch normalization.
    Args:
        encoder: The encoder instance of `VGGEncoder` that is to be inverted into a decoder
    '''
    channels_in = VGGEncoder.channels_latent
    channels_out = 3

    def __init__(self, encoder):
        super(VGGDecoder, self).__init__()

        self.decoder = self._invert_(encoder)
        
    def _invert_(self, encoder):
        '''
        Args:
            encoder (ModuleList): the encoder
        Returns:
            decoder (ModuleList): the decoder obtained by "inversion" of encoder
        '''
        modules_transpose = []

        for module in reversed(encoder):
            # print("module : ", module)
            if isinstance(module, nn.Conv2d):
                kwargs = {'in_channels' : module.out_channels, 'out_channels' : module.in_channels,
                          'kernel_size' : module.kernel_size, 'stride' : module.stride,
                          'padding' : module.padding}
                module_transpose = nn.ConvTranspose2d(**kwargs)
                module_norm = nn.BatchNorm2d(module.in_channels)
                module_act = nn.ReLU(inplace=True)
                modules_transpose += [module_transpose, module_norm, module_act]

            elif isinstance(module, nn.MaxPool2d):
                kwargs = {'kernel_size' : module.kernel_size, 'stride' : module.stride,
                          'padding' : module.padding}
                module_transpose = nn.MaxUnpool2d(**kwargs)
                modules_transpose += [module_transpose]

        # Discard the final normalization and activation, so final module is convolution with bias
        modules_transpose = modules_transpose[:-2]
        # print("module_transpose : ", modules_transpose)
        return nn.ModuleList(modules_transpose)


    def forward(self, x, pool_indices):
        '''
        Args:
            x (Tensor): code tensor obtained from encoder
            pool_indices (list): Pool indices Pytorch tensors in order the pooling modules in the encoder
        Returns:
            x (Tensor): decoded image tensor
        '''
        x_current = x

        k_pool = 0
        reversed_pool_indices = list(reversed(pool_indices))

        for module_decode in self.decoder:

            # If the module is unpooling, collect the appropriate pooling indices
            if isinstance(module_decode, nn.MaxUnpool2d):
                x_current = module_decode(x_current, indices=reversed_pool_indices[k_pool])
                k_pool += 1
            else:
                x_current = module_decode(x_current)

        return x_current


################################################################################################################################################



if __name__ == "__main__":
    img_random_VGG = torch.randn(1, 3, config.IMG_HEIGH, config.IMG_WIDTH)
    img_random_VGG2 = torch.randn(1, 3, config.IMG_HEIGH, config.IMG_WIDTH)

    encoder = VGGEncoder()
    
    enc_out, pool_indices  = encoder(img_random_VGG)
    enc_out2, pool_indices2 = encoder(img_random_VGG2)
    print(enc_out.shape)
    decoder = VGGDecoder(encoder.encoder)
    dec_out = decoder(enc_out, pool_indices)
    dec_out2 = decoder(enc_out2, pool_indices2)

    # print("dec_out : \n", dec_out)

    emb = torch.cat((enc_out, enc_out2), 0)
    
    # print("shape emb : ", emb.shape) # torch.Size([2, 512, 7, 7])

    # embedding = torch.randn(config.VGG_EMBEDDING_SHAPE) 

    # print("shape embedding : ", embedding.shape) # torch.Size([1, 512, 7, 7])
