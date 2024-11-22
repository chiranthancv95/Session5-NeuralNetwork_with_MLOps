import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np

def show_images(images, title):
    """Display a batch of images"""
    grid = make_grid(images, nrow=4, padding=2, normalize=True)
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.savefig(f'augmented_{title.lower().replace(" ", "_")}.png')
    plt.close()

# Define augmentations
augmentations = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        shear=10
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Original transform
original_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def main():
    # Create output directory
    import os
    os.makedirs('augmentation_examples', exist_ok=True)

    # Load MNIST dataset
    dataset = MNIST('./data', train=True, download=True)
    
    # Select a few sample images
    n_samples = 8
    sample_indices = torch.randint(0, len(dataset), (n_samples,))
    
    # Store original and augmented images
    original_images = []
    augmented_images = []
    
    for idx in sample_indices:
        img, _ = dataset[idx]
        
        # Original image
        original_images.append(original_transform(img))
        
        # Augmented image
        augmented_images.append(augmentations(img))
    
    # Convert lists to tensors
    original_batch = torch.stack(original_images)
    augmented_batch = torch.stack(augmented_images)
    
    # Display images
    show_images(original_batch, "Original Images")
    show_images(augmented_batch, "Augmented Images")
    
    print("Augmented images have been saved as 'augmented_original_images.png' and 'augmented_augmented_images.png'")

if __name__ == "__main__":
    main() 