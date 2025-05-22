import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor, CenterCrop, Resize, Compose, Pad
import torchvision.utils as utils

from model import Generator

def test_model(model_path, test_image_path, output_path):
    # Load the trained generator model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    generator = Generator(color_mode="RGB").to(device)
    generator.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    generator.eval()

    # Read and preprocess the test image
    test_image = Image.open(test_image_path).convert('RGB')
    fixed_size = (1024, 1024)  # Define a fixed size for the images

    # Define the preprocessing operations
    transform = Compose([
        Resize(fixed_size),
        ToTensor()
    ])

    test_image = transform(test_image).unsqueeze(0).to(device)

    # Use the trained generator model to denoise the image
    with torch.no_grad():
        output_image = generator(test_image)

    # Post-process and save the denoised image
    utils.save_image(output_image, output_path)

#for single cell segmentation
if __name__ == "__main__":
    for file in os.listdir('1024/valid/noisy'):
        test_image_path = os.path.join('1024/valid/noisy', file)
        model_path = "model/segmentation/netG.pth"  # Path to the trained generator model
        # Define the output path
        if not os.path.exists('content/seg/unet-gan'):
            os.makedirs('content/seg/unet-gan')
        output_path = 'content/seg/unet-gan' + file
        # 测试模型
        test_model(model_path, test_image_path, output_path)

#for directory segmentation
'''
    for file in os.listdir('1024/valid/noisy'):
        if file.endswith('.jpeg'):
            test_model(model_path, os.path.join('1024/valid/noisy', file), os.path.join('content/seg', file))
'''