#!/usr/bin/env python3
"""
Text Alignment Script

This script performs character-level alignment between Nom and QuocNgu text using
Levenshtein distance with dictionary-based compatibility checking, and outputs
a colorized Excel file and CSV with color code columns showing alignment results.

Usage:
    python align_texts.py configs/alignment_configs/sample_alignment.yaml

Input CSV format:
    Nom,QuocNgu
    漢字,han tu
    ...

Output:
    - Excel file with colorized alignment (black=match, red=mismatch, green=OCR fallback, blue=similarity)
    - CSV file with additional color code columns (e.g., ChuQN_txt_color, rec_result_color)
    - Color codes: B=Match, R=Mismatch, G=OCR_fallback, U=Similarity
"""

import pandas as pd
import numpy as np
import argparse
import yaml
from pathlib import Path
import ast
from xlsxwriter import Workbook


def load_config(config_path):
    """Load alignment configuration from YAML file"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_translation_dict(dict_path, encoding="utf-8-sig"):
    """Load QuocNgu to SinoNom translation dictionary"""
    df = pd.read_csv(dict_path, encoding=encoding)

    # Group SinoNom characters by QuocNgu word
    trans_dict = {}
    for _, row in df.iterrows():
        quoc_ngu = str(row["QuocNgu"]).strip().lower()
        sino_nom = str(row["SinoNom"]).strip()

        if quoc_ngu not in trans_dict:
            trans_dict[quoc_ngu] = set()
        trans_dict[quoc_ngu].add(sino_nom)

    # Convert sets to lists for consistency
    return {k: list(v) for k, v in trans_dict.items()}


def load_similarity_dict(dict_path, encoding="utf-8-sig"):
    """Load SinoNom character similarity dictionary"""
    df = pd.read_csv(dict_path, encoding=encoding)
    
    # Create dictionary mapping characters to their similar characters
    similar_dict = {}
    for _, row in df.iterrows():
        input_char = str(row["Input Character"]).strip()
        similar_chars_str = str(row["Top 20 Similar Characters"]).strip()
        
        # Parse the string representation of the list
        try:
            similar_chars = ast.literal_eval(similar_chars_str)
            # Include the original character plus similar ones
            similar_dict[input_char] = [input_char] + similar_chars
        except (ValueError, SyntaxError):
            # If parsing fails, just use the original character
            similar_dict[input_char] = [input_char]
    
    return similar_dict


def is_compatible(sino_nom_char, quoc_ngu_word, trans_dict):
    """
    Check if a SinoNom character is compatible with a QuocNgu word
    based on dictionary lookup

    Args:
        sino_nom_char: Single SinoNom character from OCR result
        quoc_ngu_word: Vietnamese word from ground truth
        trans_dict: Dictionary mapping QuocNgu words to possible SinoNom characters
    """
    if not sino_nom_char or not quoc_ngu_word:
        return False

    quoc_ngu_lower = quoc_ngu_word.strip().lower()
    possible_sino_chars = trans_dict.get(quoc_ngu_lower, [])

    return sino_nom_char in possible_sino_chars


def is_similarity_compatible(sino_nom_char, quoc_ngu_word, trans_dict, similar_dict):
    """
    Check if a SinoNom character is compatible with a QuocNgu word through character similarity
    
    Args:
        sino_nom_char: Single SinoNom character from OCR result
        quoc_ngu_word: Vietnamese word from ground truth
        trans_dict: Dictionary mapping QuocNgu words to possible SinoNom characters
        similar_dict: Dictionary mapping characters to their similar characters list
    
    Returns:
        tuple: (is_compatible, replacement_char) - replacement_char is the similar char that matches
    """
    if not sino_nom_char or not quoc_ngu_word:
        return False, None
        
    quoc_ngu_lower = quoc_ngu_word.strip().lower()
    expected_sino_chars = trans_dict.get(quoc_ngu_lower, [])
    
    if not expected_sino_chars:
        return False, None
    
    # Get similar characters for the OCR result
    similar_chars = similar_dict.get(sino_nom_char, [sino_nom_char])
    
    # Find intersection between expected characters and similar characters
    matches = list(set(expected_sino_chars) & set(similar_chars))
    
    if matches:
        # Return the first match as replacement character
        return True, matches[0]
    
    return False, None


def levenshtein_align(
    sino_nom_text, quoc_ngu_text, trans_dict, costs=None, sino_nom_ocr_text=None, similar_dict=None
):
    """
    Perform alignment between SinoNom characters and QuocNgu words using Levenshtein distance
    with dictionary-based compatibility checking, optional OCR fallback, and similarity checking

    Args:
        sino_nom_text: SinoNom text string (characters without spaces)
        quoc_ngu_text: QuocNgu text string (space-separated words)
        trans_dict: Translation dictionary mapping QuocNgu words to SinoNom characters
        costs: Dictionary with alignment costs (match, mismatch, insertion, deletion)
        sino_nom_ocr_text: Optional fallback OCR text for mismatched characters
        similar_dict: Optional similarity dictionary for character matching

    Returns:
        tuple: (aligned_sino_nom, aligned_quoc_ngu, alignment_info)
    """
    if costs is None:
        costs = {"match": 0, "mismatch": 1, "insertion": 1, "deletion": 1}

    # SinoNom: individual characters
    sino_nom_chars = list(sino_nom_text.strip())

    # Optional OCR fallback characters
    sino_nom_ocr_chars = None
    if sino_nom_ocr_text:
        sino_nom_ocr_chars = list(sino_nom_ocr_text.strip())

    # QuocNgu: split by spaces to get words
    quoc_ngu_words = quoc_ngu_text.strip().split()

    m, n = len(sino_nom_chars), len(quoc_ngu_words)

    # Initialize DP matrix
    dp = np.zeros((m + 1, n + 1), dtype=int)
    backtrace = np.full((m + 1, n + 1), "", dtype=object)

    # Initialize boundaries
    for i in range(m + 1):
        dp[i][0] = i
        if i > 0:
            backtrace[i][0] = "U"  # Up (deletion of sino nom char)

    for j in range(n + 1):
        dp[0][j] = j
        if j > 0:
            backtrace[0][j] = "L"  # Left (insertion of quoc ngu word)

    # Fill DP matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sino_nom_char = sino_nom_chars[i - 1]
            quoc_ngu_word = quoc_ngu_words[j - 1]

            # Check compatibility: can this QuocNgu word translate to this SinoNom character?
            match = is_compatible(sino_nom_char, quoc_ngu_word, trans_dict)
            subst_cost = costs["match"] if match else costs["mismatch"]

            # Calculate costs for three operations
            options = [
                (dp[i - 1][j] + costs["deletion"], "U"),  # Deletion
                (dp[i][j - 1] + costs["insertion"], "L"),  # Insertion
                (dp[i - 1][j - 1] + subst_cost, "D"),  # Match/Substitution
            ]

            dp[i][j], backtrace[i][j] = min(options)

    # Backtrack to get alignment
    aligned_sino_nom = []
    aligned_quoc_ngu = []
    alignment_info = []  # Store match type: True=match, False=mismatch, "ocr_fallback"=OCR fallback match, "similarity"=similarity match

    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and backtrace[i][j] == "D":
            # Diagonal: align both SinoNom character and QuocNgu word
            sino_nom_char = sino_nom_chars[i - 1]
            quoc_ngu_word = quoc_ngu_words[j - 1]

            # Check primary compatibility
            is_match = is_compatible(sino_nom_char, quoc_ngu_word, trans_dict)

            if is_match:
                # Primary match
                aligned_sino_nom.append(sino_nom_char)
                aligned_quoc_ngu.append(quoc_ngu_word)
                alignment_info.append(True)
            elif sino_nom_ocr_chars and i - 1 < len(sino_nom_ocr_chars):
                # Check OCR fallback if primary fails
                ocr_char = sino_nom_ocr_chars[i - 1]
                is_ocr_match = is_compatible(ocr_char, quoc_ngu_word, trans_dict)

                if is_ocr_match:
                    # OCR fallback match - use OCR character
                    aligned_sino_nom.append(ocr_char)
                    aligned_quoc_ngu.append(quoc_ngu_word)
                    alignment_info.append("ocr_fallback")
                elif similar_dict:
                    # Check similarity fallback if OCR also fails
                    is_similar, replacement_char = is_similarity_compatible(sino_nom_char, quoc_ngu_word, trans_dict, similar_dict)
                    
                    if is_similar:
                        # Similarity match - use similar character
                        aligned_sino_nom.append(replacement_char)
                        aligned_quoc_ngu.append(quoc_ngu_word)
                        alignment_info.append("similarity")
                    else:
                        # All methods failed
                        aligned_sino_nom.append(sino_nom_char)
                        aligned_quoc_ngu.append(quoc_ngu_word)
                        alignment_info.append(False)
                else:
                    # No similarity dictionary available, use primary (mismatch)
                    aligned_sino_nom.append(sino_nom_char)
                    aligned_quoc_ngu.append(quoc_ngu_word)
                    alignment_info.append(False)
            elif similar_dict:
                # No OCR fallback, check similarity directly
                is_similar, replacement_char = is_similarity_compatible(sino_nom_char, quoc_ngu_word, trans_dict, similar_dict)
                
                if is_similar:
                    # Similarity match - use similar character
                    aligned_sino_nom.append(replacement_char)
                    aligned_quoc_ngu.append(quoc_ngu_word)
                    alignment_info.append("similarity")
                else:
                    # No match found
                    aligned_sino_nom.append(sino_nom_char)
                    aligned_quoc_ngu.append(quoc_ngu_word)
                    alignment_info.append(False)
            else:
                # No fallbacks available, use primary (mismatch)
                aligned_sino_nom.append(sino_nom_char)
                aligned_quoc_ngu.append(quoc_ngu_word)
                alignment_info.append(False)

            i -= 1
            j -= 1
        elif i > 0 and backtrace[i][j] == "U":
            # Up: SinoNom character with no QuocNgu match (deletion)
            aligned_sino_nom.append(sino_nom_chars[i - 1])
            aligned_quoc_ngu.append("_")
            alignment_info.append(False)  # Mismatch (deletion)
            i -= 1
        elif j > 0 and backtrace[i][j] == "L":
            # Left: QuocNgu word with no SinoNom match (insertion)
            aligned_sino_nom.append("_")
            aligned_quoc_ngu.append(quoc_ngu_words[j - 1])
            alignment_info.append(False)  # Mismatch (insertion)
            j -= 1

    # Reverse because we built the alignment backwards
    aligned_sino_nom.reverse()
    aligned_quoc_ngu.reverse()
    alignment_info.reverse()

    return aligned_sino_nom, aligned_quoc_ngu, alignment_info


def create_color_code_string(alignment_info, aligned_elements, color_codes_config):
    """
    Create color code string from alignment info using configured color codes
    """
    color_codes = []
    for info, element in zip(alignment_info, aligned_elements):
        # Include gap characters in color codes to maintain alignment
        if info == True:
            color_codes.append(color_codes_config.get("match", "B"))
        elif info == "ocr_fallback":
            color_codes.append(color_codes_config.get("ocr_fallback", "G"))
        elif info == "similarity":
            color_codes.append(color_codes_config.get("similarity", "U"))
        else:
            color_codes.append(color_codes_config.get("mismatch", "R"))
    
    return "".join(color_codes)


def create_alignment_csv(alignments, output_path, config):
    """
    Create CSV file with same structure as input but with plain text alignment results and color columns
    """
    # Get the input data to preserve original structure
    dataset_config = config["dataset"]
    base_dir = Path(__file__).parent
    input_path = base_dir / dataset_config["input_file"]

    # Read original CSV to preserve all columns
    columns_config = dataset_config.get("columns", {})
    csv_encoding = columns_config.get("encoding", "utf-8-sig")
    df_original = pd.read_csv(input_path, encoding=csv_encoding)

    # Get column names
    sinonom_column = columns_config.get("sinonom_column", "rec_result")
    quocngu_column = columns_config.get("quocngu_column", "ChuQN_txt")
    sinonom_ocr_fallback_column = columns_config.get(
        "sinonom_ocr_fallback_column", None
    )

    # Get color mapping configuration
    output_config = config.get("output", {})
    color_mapping = output_config.get("color_mapping", {})
    
    # Extract color codes from mapping with fallback defaults
    color_codes_config = {}
    default_codes = {"match": "B", "mismatch": "R", "ocr_fallback": "G", "similarity": "U"}
    
    for match_type, default_code in default_codes.items():
        if match_type in color_mapping and "code" in color_mapping[match_type]:
            color_codes_config[match_type] = color_mapping[match_type]["code"]
        else:
            color_codes_config[match_type] = default_code

    # Create alignment lookup by row index
    alignment_lookup = {}
    for alignment_data in alignments:
        row_idx = alignment_data["row_index"]
        alignment_lookup[row_idx] = alignment_data

    # Create a copy of the original dataframe
    df_output = df_original.copy()

    # Add color columns
    sinonom_color_column = f"{sinonom_column}_color"
    quocngu_color_column = f"{quocngu_column}_color"

    # New: error count columns for rec_result and SinoNom_OCR
    rec_error_column = f"{sinonom_column}_error_count"
    ocr_error_column = (
        f"{sinonom_ocr_fallback_column}_error_count"
        if sinonom_ocr_fallback_column
        else None
    )

    # Update aligned columns with plain text results and add color columns
    for row_idx in alignment_lookup:
        alignment_data = alignment_lookup[row_idx]
        aligned_sinonom = alignment_data["aligned_sinonom"]
        aligned_quoc_ngu = alignment_data["aligned_quoc_ngu"]
        alignment_info = alignment_data["alignment_info"]

        # Create plain text aligned strings (keep gaps to maintain 1-1 element matching)
        aligned_sinonom_str = "".join(aligned_sinonom)  # Character-level: "正_綿"
        aligned_quoc_ngu_str = " ".join(aligned_quoc_ngu)  # Word-level: "xin chào _ bạn"

        # Create color code strings
        sinonom_color_code = create_color_code_string(alignment_info, aligned_sinonom, color_codes_config)
        quocngu_color_code = create_color_code_string(alignment_info, aligned_quoc_ngu, color_codes_config)

        # Calculate match rate (count True, "ocr_fallback", and "similarity" as matches)
        total_chars = len(alignment_info)
        matches = sum(
            1 for info in alignment_info if info == True or info == "ocr_fallback" or info == "similarity"
        )
        match_rate = matches / total_chars if total_chars > 0 else 0

        # Update the dataframe
        df_output.at[row_idx, sinonom_column] = aligned_sinonom_str
        df_output.at[row_idx, quocngu_column] = aligned_quoc_ngu_str
        df_output.at[row_idx, sinonom_color_column] = sinonom_color_code
        df_output.at[row_idx, quocngu_color_column] = quocngu_color_code

        # Set error counts if available
        if rec_error_column:
            df_output.at[row_idx, rec_error_column] = alignment_data.get(
                "rec_error_count", ""
            )
        if ocr_error_column:
            df_output.at[row_idx, ocr_error_column] = alignment_data.get(
                "ocr_error_count", ""
            )

    # Initialize color columns for rows without alignment data
    if sinonom_color_column not in df_output.columns:
        df_output[sinonom_color_column] = ""
    if quocngu_color_column not in df_output.columns:
        df_output[quocngu_color_column] = ""

    # Ensure error columns exist for all rows
    if rec_error_column not in df_output.columns:
        df_output[rec_error_column] = ""
    if ocr_error_column and ocr_error_column not in df_output.columns:
        df_output[ocr_error_column] = ""

    # Add match rate column if not exists
    if "Match_Rate" not in df_output.columns:
        match_rates = []
        for row_idx in range(len(df_output)):
            if row_idx in alignment_lookup:
                alignment_data = alignment_lookup[row_idx]
                alignment_info = alignment_data["alignment_info"]
                total_chars = len(alignment_info)
                matches = sum(
                    1
                    for info in alignment_info
                    if info == True or info == "ocr_fallback" or info == "similarity"
                )
                match_rate = matches / total_chars if total_chars > 0 else 0
                match_rates.append(f"{match_rate:.4f}")
            else:
                match_rates.append("")
        df_output["Match_Rate"] = match_rates

    # Save to CSV
    # Reorder columns so error counts come right after Match_Rate
    if "Match_Rate" in df_output.columns:
        cols = list(df_output.columns)
        # Remove if present to reinsert
        if rec_error_column in cols:
            cols.remove(rec_error_column)
        if ocr_error_column and ocr_error_column in cols:
            cols.remove(ocr_error_column)
        # Insert after Match_Rate
        mr_idx = cols.index("Match_Rate")
        insert_idx = mr_idx + 1
        # Insert error columns with SinoNom_OCR first, then rec_result
        if ocr_error_column:
            cols.insert(insert_idx, ocr_error_column)
            cols.insert(insert_idx + 1, rec_error_column)
        else:
            cols.insert(insert_idx, rec_error_column)
        df_output = df_output[cols]

    df_output.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Alignment CSV saved to: {output_path}")
    
    # Print color codes legend using configured values with descriptions
    legend_parts = []
    for key, code in color_codes_config.items():
        if key in color_mapping and "description" in color_mapping[key]:
            description = color_mapping[key]["description"]
            legend_parts.append(f"{code}={description}")
        else:
            key_display = key.replace("_", " ").title()
            legend_parts.append(f"{code}={key_display}")
    print(f"Color codes: {', '.join(legend_parts)}")


def safe_write_rich_string(ws, row, col, fragments):
    """Write rich text formatting to Excel cell"""
    if len(fragments) < 3:
        text = "".join([t if isinstance(t, str) else "" for t in fragments])
        ws.write(row, col, text)
    else:
        ws.write_rich_string(row, col, *fragments)


def create_colorized_excel(alignments, output_path, config):
    """
    Create Excel file with same structure as input CSV but with colorized alignment columns
    """
    # Get the input data to preserve original structure
    dataset_config = config["dataset"]
    base_dir = Path(__file__).parent
    input_path = base_dir / dataset_config["input_file"]

    # Read original CSV to preserve all columns
    columns_config = dataset_config.get("columns", {})
    csv_encoding = columns_config.get("encoding", "utf-8-sig")
    df_original = pd.read_csv(input_path, encoding=csv_encoding)

    # Get column names
    sinonom_column = columns_config.get("sinonom_column", "rec_result")
    quocngu_column = columns_config.get("quocngu_column", "ChuQN_txt")
    sinonom_ocr_fallback_column = columns_config.get(
        "sinonom_ocr_fallback_column", None
    )

    # Get color mapping configuration
    output_config = config.get("output", {})
    color_mapping = output_config.get("color_mapping", {})
    
    # Extract color codes from mapping with fallback defaults
    color_codes_config = {}
    default_codes = {"match": "B", "mismatch": "R", "ocr_fallback": "G", "similarity": "U"}
    
    for match_type, default_code in default_codes.items():
        if match_type in color_mapping and "code" in color_mapping[match_type]:
            color_codes_config[match_type] = color_mapping[match_type]["code"]
        else:
            color_codes_config[match_type] = default_code

    # Create workbook with xlsxwriter (handle NaN values)
    workbook = Workbook(str(output_path), {"nan_inf_to_errors": True})
    worksheet = workbook.add_worksheet("Alignment Results")
    
    # Define color formats using color mapping with fallbacks
    def get_color_hex(match_type, default):
        if match_type in color_mapping and "hex" in color_mapping[match_type]:
            return color_mapping[match_type]["hex"]
        return default
    
    red = workbook.add_format({"font_color": get_color_hex("mismatch", "FF0000")})
    green = workbook.add_format({"font_color": get_color_hex("ocr_fallback", "008000")})
    blue = workbook.add_format({"font_color": get_color_hex("similarity", "0000FF")})
    black = workbook.add_format({"font_color": get_color_hex("match", "000000")})
    header = workbook.add_format(
        {
            "bold": True,
            "align": "center",
            "font_color": "000000",  # Just use black for headers
        }
    )

    # Set column widths
    column_widths = {
        0: 20,  # Page_ID
        1: 30,  # Image_Box_ID
        2: 50,  # Img_Box_Coordinate
        3: 30,  # SinoNom_OCR
        4: 90,  # ChuQN_txt
        5: 90,  # rec_result (will be colorized)
    }

    for col_idx, width in column_widths.items():
        worksheet.set_column(col_idx, col_idx, width)

    # Add color columns to headers
    headers = df_original.columns.tolist()
    sinonom_color_column = f"{sinonom_column}_color"
    quocngu_color_column = f"{quocngu_column}_color"
    # New: error columns
    rec_error_column = f"{sinonom_column}_error_count"
    ocr_error_column = (
        f"{sinonom_ocr_fallback_column}_error_count"
        if sinonom_ocr_fallback_column
        else None
    )
    
    # Insert color columns after their respective data columns
    if sinonom_column in headers:
        sinonom_idx = headers.index(sinonom_column)
        headers.insert(sinonom_idx + 1, sinonom_color_column)
    
    if quocngu_column in headers:
        quocngu_idx = headers.index(quocngu_column)
        headers.insert(quocngu_idx + 1, quocngu_color_column)
    
    # Write headers
    # Ensure error columns exist, we will position them near Match_Rate when writing rows
    if rec_error_column not in headers:
        headers.append(rec_error_column)
    if ocr_error_column and ocr_error_column not in headers:
        headers.append(ocr_error_column)

    for col_idx, header_name in enumerate(headers):
        worksheet.write(0, col_idx, header_name, header)

    # Create alignment lookup by row index
    alignment_lookup = {}
    for alignment_data in alignments:
        row_idx = alignment_data["row_index"]
        alignment_lookup[row_idx] = alignment_data

    # Process each row
    for row_idx, row in df_original.iterrows():
        # Create a data row with color columns included
        row_data = {}
        
        # Copy original data
        for col_name, value in row.items():
            if pd.isna(value):
                value = ""
            row_data[col_name] = value
        
        # Add color columns (empty by default)
        row_data[sinonom_color_column] = ""
        row_data[quocngu_color_column] = ""
        # Add error columns (empty by default)
        row_data[rec_error_column] = ""
        if ocr_error_column:
            row_data[ocr_error_column] = ""

        # Get alignment data if available
        if row_idx in alignment_lookup:
            alignment_data = alignment_lookup[row_idx]
            aligned_sinonom = alignment_data["aligned_sinonom"]
            aligned_quoc_ngu = alignment_data["aligned_quoc_ngu"]
            alignment_info = alignment_data["alignment_info"]

            # Update row data with aligned text (keep gaps to maintain 1-1 element matching)
            aligned_sinonom_str = "".join(aligned_sinonom)  # Character-level: "正_綿"
            aligned_quoc_ngu_str = " ".join(aligned_quoc_ngu)  # Word-level: "xin chào _ bạn"
            
            row_data[sinonom_column] = aligned_sinonom_str
            row_data[quocngu_column] = aligned_quoc_ngu_str
            
            # Add color codes
            row_data[sinonom_color_column] = create_color_code_string(alignment_info, aligned_sinonom, color_codes_config)
            row_data[quocngu_color_column] = create_color_code_string(alignment_info, aligned_quoc_ngu, color_codes_config)

            # Add error counts if present
            if rec_error_column in row_data:
                row_data[rec_error_column] = alignment_data.get("rec_error_count", "")
            if ocr_error_column and ocr_error_column in row_data:
                row_data[ocr_error_column] = alignment_data.get("ocr_error_count", "")

            # Create rich text for SinoNom column (colorized by character)
            sinonom_fragments = []
            for sino_char, match_info in zip(aligned_sinonom, alignment_info):
                # Determine color based on match type (including gap characters)
                if match_info == True:
                    color_format = black  # Primary match
                elif match_info == "ocr_fallback":
                    color_format = green  # OCR fallback match
                elif match_info == "similarity":
                    color_format = blue   # Similarity-based match
                else:
                    color_format = red    # Mismatch (including gaps)

                sinonom_fragments.extend([color_format, sino_char])

            # Create rich text for QuocNgu column (colorized by word)
            quocngu_fragments = []
            for quoc_word, match_info in zip(aligned_quoc_ngu, alignment_info):
                # Determine color based on match type (including gap characters)
                if match_info == True:
                    color_format = black  # Primary match
                elif match_info == "ocr_fallback":
                    color_format = green  # OCR fallback match
                elif match_info == "similarity":
                    color_format = blue   # Similarity-based match
                else:
                    color_format = red    # Mismatch (including gaps)

                quocngu_fragments.extend([color_format, quoc_word + " "])
        
        # Write all column data to Excel
        # Reorder for display: place error columns immediately after Match_Rate if present
        effective_headers = headers[:]
        if "Match_Rate" in effective_headers:
            # Remove error columns
            if rec_error_column in effective_headers:
                effective_headers.remove(rec_error_column)
            if ocr_error_column and ocr_error_column in effective_headers:
                effective_headers.remove(ocr_error_column)
            # Insert after Match_Rate
            mr_idx = effective_headers.index("Match_Rate")
            insert_idx = mr_idx + 1
            # Insert SinoNom_OCR first, then rec_result
            if ocr_error_column:
                effective_headers.insert(insert_idx, ocr_error_column)
                effective_headers.insert(insert_idx + 1, rec_error_column)
            else:
                effective_headers.insert(insert_idx, rec_error_column)

        for col_idx, header_name in enumerate(effective_headers):
            if header_name in [sinonom_column, quocngu_column] and row_idx in alignment_lookup:
                # Write colorized columns
                if header_name == sinonom_column:
                    safe_write_rich_string(worksheet, row_idx + 1, col_idx, sinonom_fragments)
                elif header_name == quocngu_column:
                    safe_write_rich_string(worksheet, row_idx + 1, col_idx, quocngu_fragments)
            else:
                # Write regular data (including color columns)
                worksheet.write(row_idx + 1, col_idx, row_data.get(header_name, ""))

    workbook.close()
    print(f"Colorized Excel file saved to: {output_path}")


def process_alignment(config):
    """Main processing function using configuration"""

    # Get configuration sections
    dataset_config = config["dataset"]
    dict_config = config["dictionary"]
    alignment_config = config["alignment"]
    output_config = config["output"]
    processing_config = config.get("processing", {})

    # Resolve paths
    base_dir = Path(__file__).parent
    input_csv = base_dir / dataset_config["input_file"]
    dict_path = base_dir / dict_config["translation_dict"]
    output_path = base_dir / output_config["file"]

    # Load input data with configured encoding
    columns_config = dataset_config.get("columns", {})
    csv_encoding = columns_config.get("encoding", "utf-8-sig")

    if processing_config.get("verbose", True):
        print(f"Loading input CSV: {input_csv}")

    df = pd.read_csv(input_csv, encoding=csv_encoding)

    # Get column names from config
    sinonom_column = columns_config.get("sinonom_column", "Nom")
    quocngu_column = columns_config.get("quocngu_column", "QuocNgu")
    sinonom_ocr_fallback_column = columns_config.get(
        "sinonom_ocr_fallback_column", None
    )
    id_column = columns_config.get("id_column", None)

    # Validate required columns exist
    if sinonom_column not in df.columns:
        raise ValueError(
            f"SinoNom column '{sinonom_column}' not found in CSV. Available columns: {list(df.columns)}"
        )
    if quocngu_column not in df.columns:
        raise ValueError(
            f"QuocNgu column '{quocngu_column}' not found in CSV. Available columns: {list(df.columns)}"
        )

    # Check if optional OCR column exists
    use_ocr_fallback = (
        sinonom_ocr_fallback_column and sinonom_ocr_fallback_column in df.columns
    )

    if processing_config.get("verbose", True):
        columns_info = f"SinoNom='{sinonom_column}', QuocNgu='{quocngu_column}'"
        if use_ocr_fallback:
            columns_info += (
                f", SinoNom_OCR='{sinonom_ocr_fallback_column}' (fallback enabled)"
            )
        else:
            columns_info += " (no OCR fallback)"
        print(f"Using columns: {columns_info}")

    # Load translation dictionary
    if processing_config.get("verbose", True):
        print(f"Loading translation dictionary: {dict_path}")

    dict_encoding = dict_config.get("encoding", "utf-8-sig")
    trans_dict = load_translation_dict(dict_path, dict_encoding)

    if processing_config.get("verbose", True):
        print(f"Loaded {len(trans_dict)} QuocNgu entries")

    # Load similarity dictionary if configured
    similar_dict = None
    if "similarity_dict" in dict_config:
        similarity_path = base_dir / dict_config["similarity_dict"]
        if similarity_path.exists():
            if processing_config.get("verbose", True):
                print(f"Loading similarity dictionary: {similarity_path}")
            similar_dict = load_similarity_dict(similarity_path, dict_encoding)
            if processing_config.get("verbose", True):
                print(f"Loaded {len(similar_dict)} character similarity entries")
        else:
            if processing_config.get("verbose", True):
                print(f"Warning: Similarity dictionary not found: {similarity_path}")
    else:
        if processing_config.get("verbose", True):
            print("No similarity dictionary configured")

    # Process each line
    alignments = []
    total_matches = 0
    total_chars = 0

    # Get alignment costs from config
    costs = alignment_config.get(
        "costs", {"match": 0, "mismatch": 1, "insertion": 1, "deletion": 1}
    )

    batch_size = processing_config.get("batch_size", 100)
    show_progress = processing_config.get("show_progress", True)

    # Apply quick test limit if enabled
    if config.get("quick_test", False):
        test_limit = config.get("test_limit", 10)
        df = df.head(test_limit)
        if processing_config.get("verbose", True):
            print(f"Quick test mode enabled: processing only first {test_limit} rows")

    if processing_config.get("verbose", True):
        print(f"Processing {len(df)} text pairs...")
        print(f"Using costs: {costs}")

    for idx, row in df.iterrows():
        sinonom_text = str(row[sinonom_column]).strip()
        quoc_ngu_text = str(row[quocngu_column]).strip()

        # Get optional OCR fallback text
        sinonom_ocr_text = None
        if use_ocr_fallback:
            sinonom_ocr_text = str(row[sinonom_ocr_fallback_column]).strip()
            if sinonom_ocr_text == "nan":
                sinonom_ocr_text = None

        # Get optional ID for tracking
        row_id = None
        if id_column and id_column in df.columns:
            row_id = str(row[id_column])

        if (
            not sinonom_text
            or not quoc_ngu_text
            or sinonom_text == "nan"
            or quoc_ngu_text == "nan"
        ):
            if processing_config.get("verbose", True):
                print(f"Warning: Empty text in line {idx + 1} (ID: {row_id}), skipping")
            continue

        # Perform alignment with configured costs and optional OCR fallback and similarity checking
        aligned_sinonom, aligned_quoc_ngu, alignment_info = levenshtein_align(
            sinonom_text, quoc_ngu_text, trans_dict, costs, sinonom_ocr_text, similar_dict
        )

        # Compute error count for rec_result (count only strict mismatches)
        rec_error_count = sum(1 for info in alignment_info if info == False)

        # If SinoNom_OCR column exists for this row, run a separate alignment to count errors
        ocr_error_count = None
        if use_ocr_fallback and sinonom_ocr_text:
            _, _, ocr_alignment_info = levenshtein_align(
                sinonom_ocr_text, quoc_ngu_text, trans_dict, costs, None, similar_dict
            )
            ocr_error_count = sum(1 for info in ocr_alignment_info if info == False)

        # Store alignment with optional ID
        alignment_data = {
            "sinonom_text": sinonom_text,
            "quoc_ngu_text": quoc_ngu_text,
            "aligned_sinonom": aligned_sinonom,
            "aligned_quoc_ngu": aligned_quoc_ngu,
            "alignment_info": alignment_info,
            "row_id": row_id,
            "row_index": idx,
            "rec_error_count": rec_error_count,
            "ocr_error_count": ocr_error_count,
        }
        alignments.append(alignment_data)

        # Update statistics (count True, "ocr_fallback", and "similarity" as matches)
        line_matches = sum(
            1 for info in alignment_info if info == True or info == "ocr_fallback" or info == "similarity"
        )
        line_total = len(alignment_info)
        total_matches += line_matches
        total_chars += line_total

        if show_progress and processing_config.get("verbose", True):
            ocr_fallbacks = sum(1 for info in alignment_info if info == "ocr_fallback")
            similarity_matches = sum(1 for info in alignment_info if info == "similarity")
            primary_matches = sum(1 for info in alignment_info if info == True)
            match_str = (
                f"{line_matches}/{line_total} matches ({line_matches / line_total:.1%})"
            )
            if ocr_fallbacks > 0 or similarity_matches > 0:
                details = []
                if primary_matches > 0:
                    details.append(f"Primary: {primary_matches}")
                if ocr_fallbacks > 0:
                    details.append(f"OCR: {ocr_fallbacks}")
                if similarity_matches > 0:
                    details.append(f"Similar: {similarity_matches}")
                match_str += f" [{', '.join(details)}]"
            print(f"Line {idx + 1}: {match_str}")

    # Create both Excel (with colors) and CSV (for easy reading) outputs
    output_format = output_config.get("format", "excel")

    if output_format == "excel":
        # Create Excel with colors
        if processing_config.get("verbose", True):
            print(f"Creating colorized Excel output...")
        create_colorized_excel(alignments, output_path, config)

        # Also create CSV version
        csv_output_path = str(output_path).replace(".xlsx", ".csv")
        if processing_config.get("verbose", True):
            print(f"Creating CSV alignment output...")
        create_alignment_csv(alignments, csv_output_path, config)
    else:
        # Create CSV
        if processing_config.get("verbose", True):
            print(f"Creating CSV alignment output...")
        create_alignment_csv(alignments, output_path, config)

        # Also create Excel version if possible
        if str(output_path).endswith(".csv"):
            excel_output_path = str(output_path).replace(".csv", ".xlsx")
            if processing_config.get("verbose", True):
                print(f"Creating colorized Excel output...")
            create_colorized_excel(alignments, excel_output_path, config)

    # Print final statistics
    overall_accuracy = total_matches / total_chars if total_chars > 0 else 0
    print(f"\nAlignment Results:")
    print(f"Total characters processed: {total_chars}")
    print(f"Total matches: {total_matches}")
    print(f"Overall accuracy: {overall_accuracy:.2%}")
    print(f"Mismatch rate: {(1 - overall_accuracy):.2%}")


def main():
    parser = argparse.ArgumentParser(
        description="Align Nom and QuocNgu text using Levenshtein distance with dictionary lookup"
    )
    parser.add_argument("config_path", help="Path to alignment configuration YAML file")

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config_path)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    try:
        config = load_config(config_path)
        print(f"Loaded configuration: {config['dataset']['name']}")

        # Validate required paths exist
        base_dir = Path(__file__).parent
        input_path = base_dir / config["dataset"]["input_file"]
        dict_path = base_dir / config["dictionary"]["translation_dict"]

        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            return 1

        if not dict_path.exists():
            print(f"Error: Dictionary file not found: {dict_path}")
            return 1

        # Generate output path with suffix pattern from config
        csv_stem = input_path.stem
        dataset_dir = input_path.parent

        # Get output suffix from config
        output_suffix = config["output"].get("suffix", "_align")

        # Determine output extension based on format
        output_format = config["output"].get("format", "csv")
        extension = ".xlsx" if output_format == "excel" else ".csv"
        output_filename = f"{csv_stem}{output_suffix}{extension}"
        output_path = dataset_dir / output_filename

        # Update config with new output path
        config["output"]["file"] = str(output_path)

        print(f"Input: {input_path}")
        print(f"Output: {output_path}")

        # Process alignment
        process_alignment(config)
        print(f"Alignment completed successfully!")
        return 0

    except Exception as e:
        print(f"Error during processing: {e}")
        return 1


if __name__ == "__main__":
    exit(main())