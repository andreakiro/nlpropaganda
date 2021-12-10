from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, input_size*2),
            nn.ReLU(),
            nn.Linear(input_size*2, input_size),
            nn.ReLU(),
            nn.Linear(input_size, int(input_size*0.6)),
            nn.ReLU(),
            nn.Linear(int(input_size*0.6), int(input_size*0.2)),
            nn.ReLU(),
            nn.Linear(int(input_size*0.2), output_size)
        )

    def forward(self, x):
        return self.layers(x)
