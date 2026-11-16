import torch
from model import ConvModel
from dataset import get_dataloaders

print("CUDA available:", torch.cuda.is_available())

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)
model = ConvModel().to(device)
model.eval()

x = torch.randn(64, 3, 32, 32, device=device)

with torch.no_grad():
    output = model(x)

print(output.shape)

get_dataloaders(batch_size=32)
