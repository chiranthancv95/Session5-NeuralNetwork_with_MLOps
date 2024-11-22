import os
import torch
from datetime import datetime
from mnist_model import LightMNIST

def save_model(model, accuracy):
    """Save the trained model with timestamp and accuracy"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    filename = f"model_{timestamp}_{device}_acc{accuracy:.2f}.pth"
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), os.path.join("models", filename))
    print(f"Model saved as {filename}")

if __name__ == "__main__":
    # Load and train model
    model = LightMNIST()
    # Add your training code here
    accuracy = 95.82  # Replace with actual accuracy
    save_model(model, accuracy) 