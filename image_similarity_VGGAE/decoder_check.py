from matplotlib import image
import torch
import torchvision.transforms as transforms
import model
import config 
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Encoder, Decoder 생성 

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # TEST_IMAGE_PATH = "../input/korean-ReID/INDOOR/H00280/SN4/090404/IN_H00280_SN4_090404_27438.png"
    TEST_IMAGE_PATH = "..\input/korean-ReID/INDOOR3/H00803/SN3/102508/IN_H00803_SN3_102508_11308.png"
    image = Image.open(TEST_IMAGE_PATH)
    # image.show()
    # image.save('orgin.jpg','JPEG') 

    Encoder = model.VGGEncoder()
    Decoder = model.VGGDecoder(Encoder.encoder)
    # Load the state dict of encoder
    Encoder.load_state_dict(torch.load(config.ENCODER_MODEL_PATH, map_location=device))

    Encoder.eval()
    Encoder.to(device)
    
    Decoder.load_state_dict(torch.load(config.DECODER_MODEL_PATH, map_location=device))
    Decoder.eval()
    Decoder.to(device)

    
    with torch.no_grad():
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(config.IMG_HEIGH, config.IMG_WIDTH))])
        ae_test_img = transform(Image.open(TEST_IMAGE_PATH))
        image_tensor = ae_test_img.unsqueeze(0)
        enc_output, pooling_indices = Encoder(image_tensor)
        print(enc_output)
        dec_output = Decoder(enc_output, pooling_indices).cpu()
        dec_output = dec_output.squeeze() 
        dec_output_img = transforms.ToPILImage()(dec_output)

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(image)
        ax1.set_title("original")
        ax1.axis("off")

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(dec_output_img)
        ax2.set_title("reconstructed")
        ax2.axis("off")

        plt.show()
        # dec_output_img.save('VGG_AE_reconstruction.jpg','JPEG') 
        
        