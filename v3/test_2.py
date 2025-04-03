from data_2 import TemporalDataset
from tqdm import tqdm
import json

# Load configuration from specified path
with open('v3/config.json') as f:
    config = json.load(f)
    
dataset = TemporalDataset(config)

dataset.split = 'train'

total = 0
positive_count = 0
negative_count = 0

iteration = 0

for data in tqdm(dataset):
    total += data['paths'].shape[0]
    positive_count += data['labels'].sum()
    negative_count += data['labels'].shape[0] - data['labels'].sum()
    
    iteration += 1
    
    if iteration % 100 == 0:
        print(f'[Iteration #{iteration}] Total number of samples: {total}, Positive: {positive_count}, Negative: {negative_count}')
    
print(f'[Final] Total number of samples: {total}, Positive: {positive_count}, Negative: {negative_count}')