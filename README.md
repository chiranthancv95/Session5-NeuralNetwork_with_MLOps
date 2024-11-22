
## Model Architecture

The model uses a lightweight CNN architecture:
- Input Layer: 28x28 grayscale images
- First Block: 5x5 convolution (20 filters) + BatchNorm + ReLU + MaxPool
- Second Block: 3x3 convolution (16 filters) + BatchNorm + ReLU + MaxPool
- Output Layer: Fully connected layer to 10 classes

Key Features:
- Parameters: <25,000
- Training Time: Single epoch
- Accuracy: >95% on training data
- Input: 28x28 MNIST images
- Output: 10 classes (digits 0-9)

## MLOps Pipeline

### Automated Testing
The pipeline automatically tests:
1. Model parameter count (<25,000)
2. Input shape compatibility (28x28)
3. Output shape verification (10 classes)
4. Training accuracy (>95%)

### Continuous Integration
- Automated tests run on every push and pull request
- GitHub Actions workflow validates model performance
- Model artifacts are saved after successful training

### Deployment
- Models are saved with timestamps and accuracy metrics
- Naming format: `model_YYYYMMDD_HHMMSS_device_accXX.XX.pth`

## Setup and Installation

### Local Development

#### 1. Clone the repository:
bash
git clone https://github.com/your-username/your-repo-name.git
cd Session5

#### 2. Create and activate virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate


#### 3. Install dependencies:
pip install torch torchvision tqdm pytest pytest-cov


#### 4. Run tests:
pytest tests/test_model.py -v

#### 5. Train and deploy model:
python deploy.py

### GitHub Actions

The CI/CD pipeline automatically runs when:
- Pushing to main/master branch
- Creating a pull request
- Manually triggering the workflow

## Results

- Training Accuracy: >95% in single epoch
- Parameter Count: ~12,000
- Training Time: ~2-3 minutes on CPU