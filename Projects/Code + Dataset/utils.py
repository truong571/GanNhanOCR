import torch
from xlsxwriter.workbook import Workbook
from data import ImageDataset
from PIL import Image, ImageDraw, ImageFont
import os.path as osp
from fontTools.ttLib import TTFont
import csv
from config import (
    FONT_FAMILY,
    IMAGE_DATASET_ROOT,
    CHAR_DATASET
)

def choose_font(char: str, font_family: list[tuple[set[str], ImageFont.FreeTypeFont]]) -> ImageFont.FreeTypeFont:
    for chars, font in font_family:
        if char in chars:
            return font
    assert False, f"Character {char} not found in any font"

def char2image(
        chinese_character: str, 
        size: tuple[int, int], 
        font_size: int, 
        font_family: list[tuple[set[str], ImageFont.FreeTypeFont]]) -> Image:
    # Create a blank image
    image_width, image_height = size
    image = Image.new("RGB", (image_width,image_height), "white")
    draw = ImageDraw.Draw(image)

    # Choose a font
    font = choose_font(chinese_character, font_family)

    # Measure text size
    # text_width, text_height = draw.textsize(chinese_character, font=font)
    text_width = draw.textlength(chinese_character, font=font)
    text_height = font_size

    # Calculate position to center the text
    x = (image_width - text_width) / 2
    y = (image_height - text_height) / 2

    # Draw the text at the calculated position
    draw.text((x, y), chinese_character, font=font, fill="black")


    # Save or show the image
    # image.save("chinese_text.png")
    return image

def unicode2char(unicode: str) -> str:
    return chr(int(unicode, 16))

def unicode2image(
        unicode: str, 
        size: tuple[int, int], 
        font_size: int,
        font_family: list[tuple[set[str], ImageFont.FreeTypeFont]]) -> Image:
    return char2image(unicode2char(unicode), size, font_size, font_family)

def save_image(image: Image, path: str) -> None:
    image.save(
        osp.join(path),
        format="JPEG", quality=95, optimize=True, progressive=True
    )

def build_char_image_dataset(
        sinonom_chars: list[str], 
        dest_dir: str, 
        size: tuple[int, int], 
        font_size: int,
        font_family: list[tuple[set[str], ImageFont.FreeTypeFont]]) -> None:
    for id,c in enumerate(sinonom_chars):
        if id % 100 == 0:
            print(f"Processing {id}/{len(sinonom_chars)}")
        img = char2image(c, size, font_size, font_family)
        code = hex(ord(c))[2:].upper()
        save_image(img, osp.join(dest_dir, code + '.jpg'))

def build_unicode_image_dataset(
        sinonom_unicodes: list[str], 
        dest_dir: str, 
        size: tuple[int, int], 
        font_size: int,
        font_family: list[tuple[set[str], ImageFont.FreeTypeFont]]) -> None:
    for c in sinonom_unicodes:
        img = unicode2image(c, size, font_size, font_family)
        save_image(img, osp.join(dest_dir, c + '.jpg'))

def create_similarity_matrix(embeddings, save_path = None) -> torch.Tensor:
    cosine_similarity = torch.matmul(embeddings, embeddings.T)
    if save_path:
        print(f"Saving to {save_path}...")
        torch.save(cosine_similarity, save_path)
        print("Saved successfully!")
    return cosine_similarity

def save_topk_to_xlsx(dataset: ImageDataset, cosine_similarity: torch.Tensor, topk: int, save_path: str) -> None:
    top_k_values, top_k_indices = torch.topk(cosine_similarity, k=topk+1, dim=1)

    sino_sim = []
    for i in range(len(top_k_indices)):
        sino_sim.append([dataset.char_list[i], [dataset.char_list[j] for j in top_k_indices[i][1:]]])

    print(f"Saving to {save_path}...")
    workbook       = Workbook(save_path)
    default_format = workbook.add_format({'color': 'black', 'font_name': 'Nom Na Tong'})

    worksheet = workbook.add_worksheet("Sheet1")

    worksheet.write(0, 0, "Input Character")
    worksheet.set_column(0, 0, 5, default_format)
    worksheet.write(0, 1, f"Top {topk} Similar Characters")
    worksheet.set_column(1, 1, 50, default_format)

    for excel_row, (char, sim_chars) in enumerate(sino_sim):
        excel_row += 1
        worksheet.write(excel_row, 0, char, default_format)
        worksheet.write(excel_row, 1, sim_chars.__str__(), default_format)

    workbook.close()

    print("Saved successfully!")
    print(f"Saving to {save_path.replace('.xlsx', '.csv')}...")
    with open(save_path.replace(".xlsx", ".csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["char", f"sim"])
        for char, sim_chars in sino_sim:
            writer.writerow([char, sim_chars])
    print("Saved successfully!")

def get_characters_from_ttf(font_path: str) -> set:
    chars = set()
    with TTFont(font_path, 0, ignoreDecompileErrors=True) as ttf:
        for x in ttf["cmap"].tables:
            for (code, _) in x.cmap.items():
                chars.add(chr(code))
    return chars

def build_dataset():
    image_size = (256, 256)
    font_size = 200
    font_chars = [get_characters_from_ttf(font) for font in FONT_FAMILY]
    font_family = [(chars, ImageFont.truetype(font, font_size)) for chars, font in zip(font_chars, FONT_FAMILY)]

    with open(CHAR_DATASET, "r") as f:
        sinonom_chars = list(f.read().strip('\n '))
    build_char_image_dataset(sinonom_chars, IMAGE_DATASET_ROOT, image_size, font_size, font_family)

if __name__ == '__main__':
    build_dataset()