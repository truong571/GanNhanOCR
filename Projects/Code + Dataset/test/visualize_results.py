import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp
import re

dir_path = osp.dirname(osp.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', type=str, default='./results')

args = parser.parse_args()
result_dir = args.result_dir

# Load and sort the data
# df = pd.read_csv('results/summary.csv')
df = pd.read_csv(osp.join(dir_path, result_dir, 'summary.csv'))
df = df.sort_values('true_replace', ascending=True)

# Preprocess dict_path column
df['dict_path'] = df['dict_path']\
    .apply(lambda x: re.sub(r'.*simdicts.\\', '', x))\
    .apply(lambda x: re.sub(r'.csv$', '', x))\
    .apply(lambda x: x.replace('SinoNom_similar_Dic_', ''))\
    .apply(lambda x: x.replace('ver2_', ''))\
    .apply(lambda x: x.replace('_formatted', ''))

# Define the total sum
sum_total = 300

# Calculate the remaining portion
df['remaining'] = sum_total - (df['true_replace'] + df['false_replace'])

# Plot setup
fig, ax = plt.subplots(figsize=(10, 7))

# Horizontal stacked bar plot with upbeat colors and adjusted bar lengths
scaling_factor = 0.5
bars = ax.barh(df['dict_path'], df['true_replace'] * scaling_factor, color='#77DD77', label='True Replace')  # Green
ax.barh(df['dict_path'], df['false_replace'] * scaling_factor, left=df['true_replace'] * scaling_factor, color='#FF6961', label='False Replace')  # Red
ax.barh(df['dict_path'], df['remaining'] * scaling_factor, left=(df['true_replace'] + df['false_replace']) * scaling_factor, color='#CFCFC4', label='Remaining')  # Gray

# Add annotations
for i, (true_val, false_val, remaining_val) in enumerate(zip(df['true_replace'], df['false_replace'], df['remaining'])):
    ax.text(true_val * scaling_factor / 2, i, str(true_val), va='center', ha='center', color='black', fontsize=8)
    ax.text((true_val * scaling_factor) + (false_val * scaling_factor / 2), i, str(false_val), va='center', ha='center', color='black', fontsize=8)
    ax.text((true_val + false_val) * scaling_factor + (remaining_val * scaling_factor / 2), i, str(remaining_val), va='center', ha='center', color='black', fontsize=8)

# Add labels, title, and legend
ax.set_xlabel('Count')
ax.set_title('Similarity Dictionary Performance')
ax.legend(loc='upper right')

# Adjust layout to prevent bar overflow
plt.tight_layout()
# plt.savefig('results/summary.png')
# plt.savefig('results/summary.pdf', format='pdf')
plt.savefig(osp.join(dir_path, result_dir, 'summary.png'))
plt.savefig(osp.join(dir_path, result_dir, 'summary.pdf'), format='pdf')
plt.show()
