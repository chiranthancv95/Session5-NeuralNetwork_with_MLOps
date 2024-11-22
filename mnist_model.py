import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

class LightMNIST(nn.Module):
    def __init__(self):
        super(LightMNIST, self).__init__()
        # First block - Feature extraction
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)  # 24x24x20
        self.bn1 = nn.BatchNorm2d(20)
        self.pool1 = nn.MaxPool2d(2, 2)  # 12x12x20
        
        # Second block - Feature refinement
        self.conv2 = nn.Conv2d(20, 16, kernel_size=3)  # 10x10x16
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)  # 5x5x16

        # Classifier
        self.fc1 = nn.Linear(5 * 5 * 16, 10)

    def forward(self, x):
        # First block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Second block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Classifier
        x = x.view(-1, 5 * 5 * 16)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer):
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
    return accuracy

def main():
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

    # Initialize model
    model = LightMNIST().to(device)
    
    # Optimizer with higher learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params}")
    print("\nModel architecture:")
    print(model)

    # Train for one epoch
    print("\nTraining for 1 epoch...")
    accuracy = train(model, device, train_loader, optimizer)
    print(f"\nFinal Training accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    main() 