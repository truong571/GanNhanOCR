#%%
from PIL import Image
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random 
import torch
import torchvision.transforms as T

from config import IMAGE_DATASET_ROOT
from data import ImageDataset
from model import get_model
from preprocess import get_preprocess
from utils import create_similarity_matrix, save_topk_to_xlsx

def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call this function at the beginning of your script
set_deterministic()


def create_embedding(model, dataset, device, save_path = None) -> torch.Tensor:
    """
    Create an index that contains all of the images in the specified list of files.
    """
    all_embeddings = []
    
    with torch.no_grad():
      for data in tqdm(dataset):
        imgs = data["img"]
        embedding = []
        for img in imgs:
            embedding.append(model(img.to(device)).detach().cpu())
        all_embeddings.append(torch.cat(embedding, dim=0))

    all_embeddings = torch.stack(all_embeddings)

    if save_path:
        print(f"Saving to {save_path}...")
        torch.save(all_embeddings, save_path)
        print("Saved successfully!")

    return all_embeddings

def main(model, dataset, device, args):
    if args.load_embeddings:
        print("Loading embeddings...")
        embeddings = torch.load(args.load_embeddings)
    else: 
        print("Creating embeddings...")
        embeddings = create_embedding(model, dataset, device, args.output_embeddings)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    print("Embeddings created successfully!")
    print("Creating similarity matrix...")
    similarity_matrix = create_similarity_matrix(embeddings, args.output_similarity)
    print("Similarity matrix created successfully!")
    if args.output_xlsx:
        save_topk_to_xlsx(dataset, similarity_matrix, 20, args.output_xlsx)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--output_embeddings", type=str, required=False)
    parser.add_argument("--output_similarity", type=str, required=False)
    parser.add_argument("--output_xlsx", type=str, required=False)
    parser.add_argument("--load_embeddings", type=str, required=False)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("Model:", args.model)
    print("Output embeddings:", args.output_embeddings)
    print("Output similarity:", args.output_similarity)
    print("Output xlsx:", args.output_xlsx)
    print("Load embeddings:", args.load_embeddings)
    print("Topk:", args.topk)
    print("Device:", args.device)

    model = get_model(args.model, args.device) 
    transform_image = get_preprocess(args.model)
    dataset = ImageDataset(root=IMAGE_DATASET_ROOT, transform=transform_image)

    main(model, dataset, args.device, args)

# python3 inference.py --model dinov2_vits14 --output_embeddings embeddings_dinov2_vits14_256.pt --output_similarity cosine_similarity_dinov2_vits14_256.pt --output_xlsx SinoNom_similar_Dic_ver2_dinov2_vits14_256.xlsx --device cuda:1
# python3 inference.py --model dinov2_vitb14 --output_embeddings embeddings_dinov2_vitb14_256.pt --output_similarity cosine_similarity_dinov2_vitb14_256.pt --output_xlsx SinoNom_similar_Dic_ver2_dinov2_vitb14_256.xlsx --device cuda:1
# python3 inference.py --model dinov2_vitl14 --output_embeddings embeddings_dinov2_vitl14_256.pt --output_similarity cosine_similarity_dinov2_vitl14_256.pt --output_xlsx SinoNom_similar_Dic_ver2_dinov2_vitl14_256.xlsx --device cuda:1
# python3 inference.py --model dinov2_vitg14 --output_embeddings embeddings_dinov2_vitg14_256.pt --output_similarity cosine_similarity_dinov2_vitg14_256.pt --output_xlsx SinoNom_similar_Dic_ver2_dinov2_vitg14_256.xlsx --device cuda:1

# python3 inference.py --model resnet18 --output_embeddings embeddings_resnet18.pt --output_similarity cosine_similarity_resnet18.pt --output_xlsx SinoNom_similar_Dic_ver2_resnet18.xlsx --device cuda:1
# python3 inference.py --model resnet34 --output_embeddings embeddings_resnet34.pt --output_similarity cosine_similarity_resnet34.pt --output_xlsx SinoNom_similar_Dic_ver2_resnet34.xlsx --device cuda:1
# python3 inference.py --model resnet50 --output_embeddings embeddings_resnet50.pt --output_similarity cosine_similarity_resnet50.pt --output_xlsx SinoNom_similar_Dic_ver2_resnet50.xlsx --device cuda:1
# python3 inference.py --model resnet101 --output_embeddings embeddings_resnet101.pt --output_similarity cosine_similarity_resnet101.pt --output_xlsx SinoNom_similar_Dic_ver2_resnet101.xlsx --device cuda:1
# python3 inference.py --model resnet152 --output_embeddings embeddings_resnet152.pt --output_similarity cosine_similarity_resnet152.pt --output_xlsx SinoNom_similar_Dic_ver2_resnet152.xlsx --device cuda:1

# python3 inference.py --model resnet18 --output_embeddings embeddings_resnet18_256.pt --output_similarity cosine_similarity_resnet18_256.pt --output_xlsx SinoNom_similar_Dic_ver2_resnet18_256.xlsx --device cuda:1
# python3 inference.py --model resnet34 --output_embeddings embeddings_resnet34_256.pt --output_similarity cosine_similarity_resnet34_256.pt --output_xlsx SinoNom_similar_Dic_ver2_resnet34_256.xlsx --device cuda:1
# python3 inference.py --model resnet50 --output_embeddings embeddings_resnet50_256.pt --output_similarity cosine_similarity_resnet50_256.pt --output_xlsx SinoNom_similar_Dic_ver2_resnet50_256.xlsx --device cuda:1
# python3 inference.py --model resnet101 --output_embeddings embeddings_resnet101_256.pt --output_similarity cosine_similarity_resnet101_256.pt --output_xlsx SinoNom_similar_Dic_ver2_resnet101_256.xlsx --device cuda:1
# python3 inference.py --model resnet152 --output_embeddings embeddings_resnet152_256.pt --output_similarity cosine_similarity_resnet152_256.pt --output_xlsx SinoNom_similar_Dic_ver2_resnet152_256.xlsx --device cuda:1

# python3 inference.py --model resnet18 --output_embeddings embeddings_resnet18_256_parts.pt --output_similarity cosine_similarity_resnet18_256_parts.pt --output_xlsx SinoNom_similar_Dic_ver2_resnet18_256_parts.xlsx --device cuda:1
# python3 inference.py --model resnet34 --output_embeddings embeddings_resnet34_256_parts.pt --output_similarity cosine_similarity_resnet34_256_parts.pt --output_xlsx SinoNom_similar_Dic_ver2_resnet34_256_parts.xlsx --device cuda:1
# python3 inference.py --model resnet50 --output_embeddings embeddings_resnet50_256_parts.pt --output_similarity cosine_similarity_resnet50_256_parts.pt --output_xlsx SinoNom_similar_Dic_ver2_resnet50_256_parts.xlsx --device cuda:1
# python3 inference.py --model resnet101 --output_embeddings embeddings_resnet101_256_parts.pt --output_similarity cosine_similarity_resnet101_256_parts.pt --output_xlsx SinoNom_similar_Dic_ver2_resnet101_256_parts.xlsx --device cuda:1
# python3 inference.py --model resnet152 --output_embeddings embeddings_resnet152_256_parts.pt --output_similarity cosine_similarity_resnet152_256_parts.pt --output_xlsx SinoNom_similar_Dic_ver2_resnet152_256_parts.xlsx --device cuda:1

# python3 inference.py --model resnet18 --output_embeddings embeddings_resnet18_256_parts_2.pt --output_similarity cosine_similarity_resnet18_256_parts_2.pt --output_xlsx SinoNom_similar_Dic_ver2_resnet18_256_parts_2.xlsx --device cuda:1

# python3 inference.py --model resnet34 --output_embeddings embeddings_resnet34_256_parts_2.pt --output_similarity cosine_similarity_resnet34_256_parts_2.pt --output_xlsx SinoNom_similar_Dic_ver2_resnet34_256_parts_2.xlsx --device cuda:1
# python3 inference.py --model resnet50 --output_embeddings embeddings_resnet50_256_parts_2.pt --output_similarity cosine_similarity_resnet50_256_parts_2.pt --output_xlsx SinoNom_similar_Dic_ver2_resnet50_256_parts_2.xlsx --device cuda:1
# python3 inference.py --model resnet101 --output_embeddings embeddings_resnet101_256_parts_2.pt --output_similarity cosine_similarity_resnet101_256_parts_2.pt --output_xlsx SinoNom_similar_Dic_ver2_resnet101_256_parts_2.xlsx --device cuda:1
# python3 inference.py --model resnet152 --output_embeddings embeddings_resnet152_256_parts_2.pt --output_similarity cosine_similarity_resnet152_256_parts_2.pt --output_xlsx SinoNom_similar_Dic_ver2_resnet152_256_parts_2.xlsx --device cuda:1
    