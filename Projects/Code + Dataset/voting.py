#%%
import argparse
import csv
import os
from collections import defaultdict, Counter
from tqdm import tqdm
import glob
import json
#%%

def vote_freq_avgrank(sim_chars, topk, *wargs, **kwargs):
    freq_counter = Counter(sim_chars)
    rank_sums = defaultdict(int)
    for rank,sim_char in enumerate(sim_chars):
        rank_sums[sim_char] += rank % topk + 1
    sorted_sim_chars = sorted(freq_counter.keys(), key=lambda x: (-freq_counter[x], rank_sums[x] / freq_counter[x]))
    return sorted_sim_chars[:20]

def vote_freq_avgrank_weighted(sim_chars, topk, weights, *wargs, **kwargs):
    freq_counter = defaultdict(int)
    for pos,sim_char in enumerate(sim_chars):
        if sim_char not in freq_counter:
            freq_counter[sim_char] = 0
        freq_counter[sim_char] += weights[pos // topk]
    rank_sums = defaultdict(int)
    for rank,sim_char in enumerate(sim_chars):
        rank_sums[sim_char] += (rank % topk + 1) * weights[rank // topk]
    sorted_sim_chars = sorted(freq_counter.keys(), key=lambda x: (-freq_counter[x], rank_sums[x] / freq_counter[x]))
    return sorted_sim_chars[:20]

def vote_avgrank(sim_chars, topk, *wargs, **kwargs):
    freq_counter = Counter(sim_chars)
    rank_sums = defaultdict(int)
    for rank,sim_char in enumerate(sim_chars):
        rank_sums[sim_char] += rank % topk + 1
    sorted_sim_chars = sorted(freq_counter.keys(), key=lambda x: rank_sums[x] / freq_counter[x])
    return sorted_sim_chars[:20]

def vote_avgrank_weighted(sim_chars, topk, weights, *wargs, **kwargs):
    freq_counter = defaultdict(int)
    for pos,sim_char in enumerate(sim_chars):
        if sim_char not in freq_counter:
            freq_counter[sim_char] = 0
        freq_counter[sim_char] += weights[pos // topk]
    rank_sums = defaultdict(int)
    for rank,sim_char in enumerate(sim_chars):
        rank_sums[sim_char] += (rank % topk + 1) * weights[rank // topk]
    sorted_sim_chars = sorted(freq_counter.keys(), key=lambda x: rank_sums[x] / freq_counter[x])
    return sorted_sim_chars[:20]

def vote_sqravgrank(sim_chars, topk, *wargs, **kwargs):
    freq_counter = Counter(sim_chars)
    rank_sums = defaultdict(int)
    for rank,sim_char in enumerate(sim_chars):
        rank_sums[sim_char] += rank % topk + 1
    sorted_sim_chars = sorted(freq_counter.keys(), key=lambda x: rank_sums[x] / freq_counter[x] ** 2)
    return sorted_sim_chars[:20]

def vote_sqravgrank_weighted(sim_chars, topk, weights, *wargs, **kwargs):
    freq_counter = defaultdict(int)
    for pos,sim_char in enumerate(sim_chars):
        if sim_char not in freq_counter:
            freq_counter[sim_char] = 0
        freq_counter[sim_char] += weights[pos // topk]
    rank_sums = defaultdict(int)
    for rank,sim_char in enumerate(sim_chars):
        rank_sums[sim_char] += (rank % topk + 1) * weights[rank // topk]
    sorted_sim_chars = sorted(freq_counter.keys(), key=lambda x: rank_sums[x] / freq_counter[x] ** 2)
    return sorted_sim_chars[:20]

def read_csv(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            char = row[0]
            sim_chars = eval(row[1])
            data[char] = sim_chars
    return data

def aggregate_data(files):
    aggregated_data = defaultdict(list)
    for file in tqdm(files):
        data = read_csv(file)
        for char, sim_chars in data.items():
            aggregated_data[char].extend(sim_chars)
    return aggregated_data

def compute_rankings(aggregated_data, vote_method, topk, weights=None):
    if isinstance(vote_method, str):
        vote_method = globals()[vote_method]
    final_rankings = {}
    for char, sim_chars in tqdm(aggregated_data.items()):
        final_rankings[char] = vote_method(sim_chars, topk, weights=weights)
    return final_rankings
#%%
def write_csv(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['char', 'sim'])
        for char, sim_chars in data.items():
            writer.writerow([char, str(sim_chars)])

def main(files, output_file, vote_method, topk):
    print(f"Aggregating data from")
    for file in files:
        print(file)
    print()
    aggregated_data = aggregate_data(files)
    print("Data aggregated successfully!")
    print("Computing rankings...")
    final_rankings = compute_rankings(aggregated_data, vote_method, topk)
    print("Rankings computed successfully!")
    print(f"Writing rankings to {output_file}...")
    write_csv(output_file, final_rankings)
    print("Rankings written successfully!")
#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="archived/results")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--method", type=str, default="vote_freq_avgrank")
    args = parser.parse_args()

    voting_models = [
        "SinoNom_similar_Dic_ver2_dinov2_vitg14_256_parts",
        "SinoNom_similar_Dic_ver2_dinov2_vitb14_256_parts",
        "SinoNom_similar_Dic_ver2_dinov2_vitl14_256_parts",
        "SinoNom_similar_Dic_ver2_dinov2_vits14_256_parts",
        # "SinoNom_similar_Dic_ver2_dinov2_vitg14_256_parts_2",
        # "SinoNom_similar_Dic_ver2_dinov2_vitb14_256_parts_2",
        # "SinoNom_similar_Dic_ver2_dinov2_vitl14_256_parts_2",
        # "SinoNom_similar_Dic_ver2_dinov2_vits14_256_parts_2",
    ]

    with open(args.output.replace('.csv', '.json'), 'w') as f:
        json.dump(voting_models, f, indent=4)

    if len(voting_models) > 1:
        files = [os.path.join(args.input_dir, f"{model}.csv") for model in voting_models]
    else:
        files = glob.glob(os.path.join(args.input_dir, '*.csv'))

    main(files, args.output, globals()[args.method], args.topk)
#%%