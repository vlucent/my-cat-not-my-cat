#!/usr/bin/env python3

import torch
#import torch.nn as nn
from torchvision import models, transforms
#from torch.utils.data import DataLoader
#from torchvision.datasets import ImageFolder
import os

from PIL import Image



# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # Load the model (same architecture as trained)
    model = models.efficientnet_b0(pretrained=True)  # Use the same model architecture
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)  # Assuming 2 classes: your cat, other cats

    # Load the saved weights
    model.load_state_dict(torch.load('efficientnet_cat_classifier.pth'))
    model.eval()  # Set the model to evaluation mode

    # Define the transformations (same as used in training)
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    # Define the path to the folder containing new cat images
    image_folder = 'new_cat_images/'

    # Loop through each image in the folder
    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)
        
        # Open the image and apply the transformations
        image = Image.open(img_path).convert('RGB')
        image = data_transforms(image)
        image = image.unsqueeze(0)  # Add batch dimension (since model expects a batch)

        # Move the image to the same device as the model (GPU or CPU)
        image = image.to(device)
        
        # Perform the inference
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        #confidence score 
        probs = torch.nn.functional.softmax(output, dim=1)
        
        # Assuming class 0 is "other cats" and class 1 is "shami"
        if predicted.item() == 1:
            print(f"recognized as shami: '{img_name}' with confidence: {probs[0][predicted.item()]}")
        #else:
            #print(f"recognized as other cat: '{img_name}' with confidence: {probs[0][predicted.item()]}")



# Run the main function
if __name__ == "__main__":
    main()