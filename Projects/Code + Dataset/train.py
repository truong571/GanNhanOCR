#%%
import argparse
import glob
import voting
import os
import os.path as osp
from test.eval_dict import process_dict
from tqdm import tqdm
import json
import prettytable
import torch
import numpy as np
import shutil
from config import (
    schedule
)
#%%
best_model = None
best_score = 0

def evaluate(file):
    result = process_dict(file[0])
    file[1] = result['true_replace']
    return result

def create_vote(files, method, topk):
    aggregated_data = voting.aggregate_data([file[0] for file in files])
    # weights = [file[1] for file in files]
    weights = [torch.sigmoid(torch.tensor(file[1])).item() for file in files]
    final_rankings = voting.compute_rankings(aggregated_data, method, topk, weights=weights)
    return final_rankings

def softmax_fitness(files):
    fitness = torch.softmax(torch.tensor([file[1] for file in files], dtype=float), dim=0).numpy()
    return fitness

def normalized_fitness(files):
    fitness = np.array([file[1] for file in files], dtype=float)
    fitness /= fitness.sum()
    return fitness

def uniform_fitness(files):
    return np.ones(len(files)) / len(files)

def get_fitness(files, num, fitness_calc, replace):
    assert num <= len(files)
    fitness = fitness_calc(files)
    idxs = np.random.choice(len(files), size=num, p=fitness, replace=replace).tolist()
    return sorted([files[idx] for idx in idxs], key=lambda x: x[1], reverse=True)

def info_files(files):
    print('Files selected:')
    table = prettytable.PrettyTable()
    table.field_names = ['Id', 'File', 'Score']
    for id,file in enumerate(files):
        table.add_row([id] + file)
    print(table)

def train(epoch, files, p_gen, p_sel, topk, output_dir, method, fitness_calc, args):
    global best_model, best_score

    print()
    print('-'*20)
    print('Epoch:', epoch)
    print()

    info_files(files)

    old_size = len(files)

    num_generation = int(len(files) * p_gen)
    for i in range(num_generation):
        print('Epoch:', epoch, '| Generation:', i)
        print('Selecting files...')
        sub_files = get_fitness(files, int(old_size * p_sel), fitness_calc, args.replace)
        info_files(sub_files)

        print('Start voting...')
        # new_data = create_vote(sub_files, voting.vote_freq_avgrank, topk)
        new_data = create_vote(sub_files, method, topk)

        print(f'Writing to file')
        file_name = f'epoch_{epoch}_gen_{i}'
        with open(osp.join(output_dir, f'{file_name}.json'), 'w') as f:
            json.dump([file[0] for file in sub_files], f, indent=4)
        voting.write_csv(osp.join(output_dir, f'{file_name}.csv'), new_data)

        print('Evaluating...')
        files.append([osp.join(output_dir, f'{file_name}.csv'), 0])
        result = evaluate(files[-1])
        if result['true_replace'] > best_score:
            best_score = result['true_replace']
            best_model = files[-1]
            print('New best model found!')
            with open(osp.join(output_dir, 'best_model.json'), 'w') as f:
                data = {
                    'file': best_model[0],
                    'result': result
                }
                json.dump(data, f, indent=4)
        print('Result:', result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--weight", type=str, required=True)
    parser.add_argument("--replace", action='store_true')
    parser.add_argument("--filter", action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = [[file, 0] for file in glob.glob(osp.join(args.input_dir, '*.csv'))]

    print("Schedule:")
    print(json.dumps(schedule, indent=4))
    
    print('Indexing files...')
    for file in tqdm(files):
        evaluate(file)
    if args.filter:
        files = list(filter(lambda x: x[1] >= 100, files))
    files.sort(key=lambda x: x[1], reverse=True)

    fitness_calc = {
        'softmax': softmax_fitness,
        'normalized': normalized_fitness,
        'uniform': uniform_fitness
    }[args.weight]

    if args.replace is None:
        args.replace = False

    # Print info before training
    print('Number of files:', len(files))
    print('Input directory:', args.input_dir)
    print('Output directory:', args.output_dir)
    print('Topk:', args.topk)
    print('Method:', args.method)
    print('Weight:', args.weight)
    print('Replace:', args.replace)
    print('Filter:', args.filter)
    
    epoch = 0
    for i in range(len(schedule)):
        for j in range(schedule[i][0]):
            epoch += 1
            train(epoch, files, schedule[i][1], schedule[i][2], args.topk, args.output_dir, args.method, fitness_calc, args)
            print('Cutting off...')
            files.sort(key=lambda x: x[1], reverse=True)
            files = files[:int(len(files) * schedule[i][3])]
            print('Number of remaining files:', len(files))
            if len(files) <= 1:
                break

# python3 train.py --input_dir archived/results --output_dir archived/trained/train_freq_avgrank --method vote_freq_avgrank
# python3 train.py --input_dir archived/results --output_dir archived/trained/train_avgrank --method vote_avgrank
# python3 train.py --input_dir archived/results --output_dir archived/trained/train_sqravgrank --method vote_sqravgrank
# python3 train.py --input_dir archived/results --output_dir archived/trained/train_freq_avgrank_weighted --method vote_freq_avgrank_weighted
# python3 train.py --input_dir archived/results --output_dir archived/trained/train_avgrank_weighted --method vote_avgrank_weighted
# python3 train.py --input_dir archived/results --output_dir archived/trained/train_sqravgrank_weighted --method vote_sqravgrank_weighted

# train  1 -> filter, sigmoid, replace, schedule 1
# train  2 -> filter, sigmoid, replace, schedule 2
# train  3 -> no filter, sigmoid, replace, schedule 2

# train  4 -> no filter, softmax, no replace, schedule 2  
# train  5 -> filter, softmax, no replace, schedule 2
# train  6 -> no filter, softmax, replace, schedule 2  
# train  7 -> filter, softmax, replace, schedule 2

# train  8 -> no filter, proportion, no replace, schedule 2
# train  9 -> filter, proportion, no replace, schedule 2
# train 10 -> no filter, proportion, replace, schedule 2
# train 11 -> filter, proportion, replace, schedule 2

# train 12 -> no filter, sigmoid, no replace, schedule 2
# train 13 -> filter, sigmoid, no replace, schedule 2

#%%
# cnt = 0
# for weight in ['softmax', 'normalized', 'uniform']:
#     for rep in ['norep', 'replace']:
#         for filt in ['nofilt', 'filter']:
#             for method in ['freq_avgrank', 'avgrank', 'sqravgrank']:
#                 command = f'python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_{method}_{weight}_{filt}_{rep} --method vote_{method} --weight {weight}'
#                 if filt == 'filter':
#                     command += ' --filter'
#                 if rep == 'replace':
#                     command += ' --replace'
#                 cnt += 1
#                 print(cnt, '.', command)
#%%

# 1 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_freq_avgrank_softmax_nofilt_norep --method vote_freq_avgrank --weight softmax
# 2 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_avgrank_softmax_nofilt_norep --method vote_avgrank --weight softmax
# 3 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_sqravgrank_softmax_nofilt_norep --method vote_sqravgrank --weight softmax
# 4 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_freq_avgrank_softmax_filter_norep --method vote_freq_avgrank --weight softmax --filter
# 5 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_avgrank_softmax_filter_norep --method vote_avgrank --weight softmax --filter
# 6 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_sqravgrank_softmax_filter_norep --method vote_sqravgrank --weight softmax --filter
# 7 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_freq_avgrank_softmax_nofilt_replace --method vote_freq_avgrank --weight softmax --replace
# 8 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_avgrank_softmax_nofilt_replace --method vote_avgrank --weight softmax --replace
# 9 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_sqravgrank_softmax_nofilt_replace --method vote_sqravgrank --weight softmax --replace
# 10 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_freq_avgrank_softmax_filter_replace --method vote_freq_avgrank --weight softmax --filter --replace
# 11 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_avgrank_softmax_filter_replace --method vote_avgrank --weight softmax --filter --replace
# 12 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_sqravgrank_softmax_filter_replace --method vote_sqravgrank --weight softmax --filter --replace
# 13 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_freq_avgrank_normalized_nofilt_norep --method vote_freq_avgrank --weight normalized
# 14 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_avgrank_normalized_nofilt_norep --method vote_avgrank --weight normalized
# 15 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_sqravgrank_normalized_nofilt_norep --method vote_sqravgrank --weight normalized
# 16 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_freq_avgrank_normalized_filter_norep --method vote_freq_avgrank --weight normalized --filter
# 17 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_avgrank_normalized_filter_norep --method vote_avgrank --weight normalized --filter
# 18 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_sqravgrank_normalized_filter_norep --method vote_sqravgrank --weight normalized --filter
# 19 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_freq_avgrank_normalized_nofilt_replace --method vote_freq_avgrank --weight normalized --replace
# 20 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_avgrank_normalized_nofilt_replace --method vote_avgrank --weight normalized --replace
# 21 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_sqravgrank_normalized_nofilt_replace --method vote_sqravgrank --weight normalized --replace
# 22 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_freq_avgrank_normalized_filter_replace --method vote_freq_avgrank --weight normalized --filter --replace
# 23 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_avgrank_normalized_filter_replace --method vote_avgrank --weight normalized --filter --replace
# 24 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_sqravgrank_normalized_filter_replace --method vote_sqravgrank --weight normalized --filter --replace
# 25 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_freq_avgrank_uniform_nofilt_norep --method vote_freq_avgrank --weight uniform
# 26 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_avgrank_uniform_nofilt_norep --method vote_avgrank --weight uniform
# 27 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_sqravgrank_uniform_nofilt_norep --method vote_sqravgrank --weight uniform
# 28 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_freq_avgrank_uniform_filter_norep --method vote_freq_avgrank --weight uniform --filter
# 29 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_avgrank_uniform_filter_norep --method vote_avgrank --weight uniform --filter
# 30 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_sqravgrank_uniform_filter_norep --method vote_sqravgrank --weight uniform --filter
# 31 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_freq_avgrank_uniform_nofilt_replace --method vote_freq_avgrank --weight uniform --replace
# 32 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_avgrank_uniform_nofilt_replace --method vote_avgrank --weight uniform --replace
# 33 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_sqravgrank_uniform_nofilt_replace --method vote_sqravgrank --weight uniform --replace
# 34 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_freq_avgrank_uniform_filter_replace --method vote_freq_avgrank --weight uniform --filter --replace
# 35 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_avgrank_uniform_filter_replace --method vote_avgrank --weight uniform --filter --replace
# 36 . python3 train.py --input_dir archived/results --output_dir archived/trained_new/train_sqravgrank_uniform_filter_replace --method vote_sqravgrank --weight uniform --filter --replace

#%%
# train_folders = glob.glob('archived/trained_new/*')
# output_dir = 'archived/best_ckpts'
# os.makedirs(output_dir, exist_ok=True)
# max_true_rep = 0
# for folder in train_folders:
#     name = osp.basename(folder).split('_', 1)[1]
#     epochs = sorted(list(map(lambda x: osp.basename(x).split('.')[0].split('_'), glob.glob(osp.join(folder, 'epoch_*')))), key=lambda x: (int(x[1]), int(x[3])), reverse=True)
#     with open(osp.join(folder, 'best_model.json')) as f:
#         result = json.load(f)
#     print("Copying", result['file'], "to", osp.join(output_dir, name + "_sche2.csv"))
#     shutil.copy(result['file'], osp.join(output_dir, name + "_sche2.csv"))
#%%