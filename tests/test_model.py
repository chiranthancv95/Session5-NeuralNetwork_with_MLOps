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
def test_model_robustness_to_noise():
    """Test if model can handle noisy inputs"""
    model = LightMNIST()
    model.eval()
    
    # Create a clean test input
    clean_input = torch.randn(1, 1, 28, 28)
    clean_output = model(clean_input)
    
    # Add Gaussian noise
    noise = torch.randn(1, 1, 28, 28) * 0.1
    noisy_input = clean_input + noise
    noisy_output = model(noisy_input)
    
    # Check if predictions are stable (same class)
    clean_pred = clean_output.argmax(dim=1)
    noisy_pred = noisy_output.argmax(dim=1)
    
    assert clean_pred == noisy_pred, "Model predictions should be stable under small noise"

def test_model_memory_efficiency():
    """Test if model's memory usage is within acceptable limits"""
    import psutil
    import os
    
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    # Create and run model
    model = LightMNIST()
    test_input = torch.randn(100, 1, 28, 28)  # Batch size of 100
    _ = model(test_input)
    
    # Get final memory usage
    final_memory = process.memory_info().rss / 1024 / 1024
    memory_used = final_memory - initial_memory
    
    # Check if memory usage is less than 500MB
    assert memory_used < 500, f"Model used {memory_used:.2f}MB of memory, should be less than 500MB"

def test_model_inference_speed():
    """Test if model inference is fast enough"""
    import time
    
    model = LightMNIST()
    model.eval()
    
    # Prepare batch of images
    batch_size = 32
    test_input = torch.randn(batch_size, 1, 28, 28)
    
    # Warm-up run
    _ = model(test_input)
    
    # Measure inference time
    start_time = time.time()
    num_runs = 10
    
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(test_input)
    
    end_time = time.time()
    avg_time_per_batch = (end_time - start_time) / num_runs
    avg_time_per_image = avg_time_per_batch / batch_size
    
    # Assert inference time is less than 1ms per image on average
    assert avg_time_per_image < 0.001, f"Inference too slow: {avg_time_per_image*1000:.2f}ms per image"
