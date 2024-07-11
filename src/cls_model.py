import torch
import torch.nn as nn
import torch.nn.functional as F


class ExampleANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(3 * 32 * 32, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ExampleCNN(nn.Module):
    """
    Reference: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#define-a-convolutional-neural-network
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)     # Input channels: 3, Output channels: 6, Kernel size: 5
        self.pool = nn.MaxPool2d(2, 2)      # Max pooling over a (2, 2) window
        self.conv2 = nn.Conv2d(6, 16, 5)    # Input channels: 6, Output channels: 16, Kernel size: 5
        self.conv3 = nn.Conv2d(16, 32, 3)   # Input channels: 16, Output channels: 32, Kernel size: 3
        self.fc1 = nn.Linear(32 * 3 * 3, 120)  # Input features: 32*3*3, Output features: 120
        self.fc2 = nn.Linear(120, 84)       # Input features: 120, Output features: 84
        self.fc3 = nn.Linear(84, 10)        # Input features: 84, Output features: 10 (for CIFAR-10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Apply conv1, then activation, then pooling
        x = self.pool(F.relu(self.conv2(x)))  # Apply conv2, then activation, then pooling
        x = F.relu(self.conv3(x))             # Apply conv3, then activation
        x = torch.flatten(x, 1)               # Flatten the tensor except batch dimension
        x = F.relu(self.fc1(x))               # Apply fc1, then activation
        x = F.relu(self.fc2(x))               # Apply fc2, then activation
        x = self.fc3(x)                       # Apply fc3 (no activation, since it's often combined with loss function)
        return x

if __name__ == "__main__":
    x = torch.randn((1, 3, 32, 32))
    model = ExampleNetwork()
    out = model(x)
    print(out.shape)