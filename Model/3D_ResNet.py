import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18

# Load the pretrained 3D ResNet model
model = r3d_18(pretrained=True)