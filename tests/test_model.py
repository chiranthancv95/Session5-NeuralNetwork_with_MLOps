import sys
import os
import torch
import pytest
from datetime import datetime
from tqdm import tqdm


# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mnist_model import LightMNIST

def test_parameter_count():
    """Test if model has less than 25000 parameters"""
    model = LightMNIST()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 25000, f"Model has {total_params} parameters, should be less than 25000"

def test_input_shape():
    """Test if model accepts 28x28 input"""
    model = LightMNIST()
    batch_size = 1
    test_input = torch.randn(batch_size, 1, 28, 28)
    try:
        output = model(test_input)
        assert True
    except:
        assert False, "Model failed to process 28x28 input"

def test_output_shape():
    """Test if model outputs 10 classes"""
    model = LightMNIST()
    batch_size = 1
    test_input = torch.randn(batch_size, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (batch_size, 10), f"Output shape is {output.shape}, should be {(batch_size, 10)}"

def test_model_accuracy():
    """Test if model achieves >95% accuracy on training data"""
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Setup device
    device = torch.device("cpu")  # Use CPU for CI/CD
# Set random seed for reproducibility
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data preprocessing and augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=128, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model and optimizer
    model = LightMNIST().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    # Train for one epoch
    model.train()
    correct = 0
    total = 0
    model.train()
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        output = model(data)
        loss = F.nll_loss(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(data)
        
        if batch_idx % 100 == 0:
            current_accuracy = 100. * correct / total
            print(f'Batch {batch_idx}, Current Accuracy: {current_accuracy:.2f}%')
    
    accuracy = 100. * correct / total
    print(f"Final Accuracy: {accuracy:.2f}%")
    
    assert accuracy > 95, f"Accuracy is {accuracy}%, should be >95%" 