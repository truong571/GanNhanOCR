#%%
import torch
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as T
import matplotlib.pyplot as plt
#%%
def __dino_transform(img: Image.Image|np.ndarray) -> torch.Tensor:
    return T.Compose([T.ToTensor(), T.Resize(244), T.CenterCrop(224), T.Normalize([0.5], [0.5])])(img)[:3].unsqueeze(0)

def __resnet_transform(img: Image.Image|np.ndarray) -> torch.Tensor:
    preprocess = T.Compose([
        T.Resize((100, 100), interpolation=cv2.INTER_LINEAR),  # Resize with linear interpolation
        T.ToTensor(),
        # T.Resize(256, interpolation=cv2.INTER_LINEAR),
        # T.CenterCrop(224),
        # T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])  # Normalization as requested
    ])
    if isinstance(img, Image.Image):
        img = img.convert('L')
    else:
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert('L')
    return preprocess(img).unsqueeze(0)

def __find_rectangle(img: np.ndarray): 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    best_rect = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            best_rect = cv2.boundingRect(contour)
    return best_rect

def __focus_crop_image(img: np.ndarray) -> np.ndarray:
    rect = __find_rectangle(img)
    return img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

def __transform_image(img: Image.Image, transform = None) -> list[torch.Tensor]:
    if transform is None:
        return [img]
    return [transform(img)]

def __transform_image_2(img: Image.Image, transform = None) -> list[torch.Tensor]:
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    width, height = img.shape[1], img.shape[0]
    img = __focus_crop_image(img)
    img_parts = [img]
    divs = [2, 3]
    for div in divs:
        for i in range(div):
            for j in range(div):
                img_part = img[int(i/div*width):int((i+1)/div*width), int(j/div*height):int((j+1)/div*height)]
                img_parts.append(img_part)
    if transform:
        img_parts = [transform(img_part) for img_part in img_parts]
    return img_parts

def __transform_image_3(img: Image.Image, transform = None) -> list[torch.Tensor]:
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    width, height = img.shape[1], img.shape[0]
    img = __focus_crop_image(img)
    img_parts = [img]
    divs = [2, 3]
    for div in divs:
        for i in range(div):
            img_parts.append(img[int(i/div*width):int((i+1)/div*width), :])
        for i in range(div):
            img_parts.append(img[:, int(i/div*height):int((i+1)/div*height)])
    if transform:
        img_parts = [transform(img_part) for img_part in img_parts]
    return img_parts

def dino_transform_image(img: Image.Image) -> list[torch.Tensor]:
    return __transform_image(img, __dino_transform)

def dino_transform_image_2(img: Image.Image) -> list[torch.Tensor]:
    return __transform_image_2(img, __dino_transform)

def dino_transform_image_3(img: Image) -> list[torch.Tensor]:
    return __transform_image_3(img, __dino_transform)

def resnet_transform_image(img: Image) -> list[torch.Tensor]:
    return __transform_image(img, __resnet_transform)

def resnet_transform_image_2(img: Image) -> list[torch.Tensor]:
    return __transform_image_2(img, __resnet_transform)

def resnet_transform_image_3(img: Image) -> list[torch.Tensor]:
    return __transform_image_3(img, __resnet_transform)

def get_preprocess(name: str):
    if "dino" in name:
        return dino_transform_image_3
    elif "resnet" in name:
        return resnet_transform_image_2
    else:
        raise ValueError("Invalid model name!")
    
#%%
# if __name__ == '__main__':
#     # img = Image.open("data/images/4E3B.jpg").convert("RGB")
#     img = Image.open("data/images/99A9.jpg").convert("RGB")
#     img_parts = __transform_image_2(img)
#     fig, axes = plt.subplots(1, len(img_parts), figsize=(10, 2))
#     axes = axes.flatten()

#     for i, img_part in enumerate(img_parts):
#         axes[i].imshow(cv2.cvtColor(img_part, cv2.COLOR_BGR2RGB))
#         axes[i].axis('off')

#     plt.tight_layout()
#     # plt.show()
#     plt.savefig("archived/stuffs/99A9_parts.svg", format="svg")
#%%