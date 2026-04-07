#%%
import torch
from torchvision import models, transforms
#%%

class DinoV2:
    def __init__(self, checkpoint: str, device: str = "cuda"):
        print(f"Loading {checkpoint}...")
        model = torch.hub.load("facebookresearch/dinov2", checkpoint)
        device = torch.device(device)
        print("Device:", device)
        model.to(device)
        model.eval()
        self.model = model
        print("Model loaded successfully!")
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(img)[0]
        
class ResNet:
    def __init__(self, checkpoint: str, device: str = "cuda"):
        if checkpoint == "resnet18":
            model = models.resnet18(pretrained=True)
        elif checkpoint == "resnet34":
            model = models.resnet34(pretrained=True)
        elif checkpoint == "resnet50":
            model = models.resnet50(pretrained=True)
        elif checkpoint == "resnet101":
            model = models.resnet101(pretrained=True)
        elif checkpoint == "resnet152":
            model = models.resnet152(pretrained=True)
        else:
            raise ValueError("Invalid checkpoint!")
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the classification head
        model.to(device)
        model.eval()
        self.model = model
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(img).squeeze()

def get_model(name: str, device: str = "cuda"):
    if "dinov2" in name:
        return DinoV2(name, device)
    elif "resnet" in name:
        return ResNet(name, device)
    