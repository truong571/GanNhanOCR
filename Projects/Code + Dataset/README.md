# Installation

## Conda

```bash
conda env create -f dinov2.yaml
```

# Dataset (TBA)

# Usage

## Create dataset

Run `python utils.py` to create dataset of 256x256 images for `data/images/`.

## Inference

```bash
python inference.py --model dinov2_vits14 --output_embeddings embeddings_dinov2_vits14_256.pt --output_similarity cosine_similarity_dinov2_vits14_256.pt --output_xlsx SinoNom_similar_Dic_ver2_dinov2_vits14_256.xlsx --device cuda
```

## Run server for character visualizing

```bash
python hosting.py
python server.py
```

## Evaluation (TBA)