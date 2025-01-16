#!/usr/bin/env python3

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

# Define paths to your dataset

train_dir = 'dataset/train'
val_dir = 'dataset/val'

# Define hyperparameters
num_epochs = 10
batch_size = 32
learning_rate = 1e-4
num_classes = 2  # Change this based on your classification problem

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load the EfficientNet model and modify the classifier
def load_model():
    # Load pre-trained EfficientNet-B0 model
    model = models.efficientnet_b0(pretrained=True)
    
    # Modify the classifier for your specific task (binary classification in this case)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    # Move the model to the appropriate device (GPU or CPU)
    model = model.to(device)
    return model

# Function to define data transformations and load the dataset
def load_data(train_dir, val_dir, batch_size):
    # Define image transformations
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = ImageFolder(train_dir, transform=data_transforms)
    val_dataset = ImageFolder(val_dir, transform=data_transforms)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# Function to define the training process
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update loss and accuracy statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Print training stats for each epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

# Function to evaluate the model on the validation set
def evaluate_model(model, val_loader, criterion):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader.dataset)
    val_acc = correct / total
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')

# Main function to load data, model, and start training
def main():
    # Load the model
    model = load_model()

    # Load the dataset
    train_loader, val_loader = load_data(train_dir, val_dir, batch_size)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    # Evaluate the model
    evaluate_model(model, val_loader, criterion)

    # Save the trained model
    torch.save(model.state_dict(), 'efficientnet_cat_classifier_v2.pth')
    print("Model saved to efficientnet_cat_classifier_v2.pth")

# Run the main function
if __name__ == "__main__":
    main()
