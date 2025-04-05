from data_2 import TemporalDataset
from tqdm import tqdm
import json

# Load configuration from specified path
with open('v3/config.json') as f:
    config = json.load(f)
    
dataset = TemporalDataset(config, k=3, m_d=100)

dataset.split = 'test'

total = 0
positive_count = 0
negative_count = 0

iteration = 0

for data in tqdm(dataset):
    total += data['paths'].shape[0]
    positive_count += data['labels'].sum()
    negative_count += data['labels'].shape[0] - data['labels'].sum()
    
    data['labels'] = data['labels'].squeeze(1)
    
    print(data['masks'][data['labels'] == 1])
    print(data['paths'][data['labels'] == 1][data['masks'][data['labels'] == 1] == 1])
    
    iteration += 1
    
    if iteration % 1 == 0:
        print(f'[Iteration #{iteration}] Total number of samples: {total}, Positive: {positive_count}, Negative: {negative_count}')
    
print(f'[Final] Total number of samples: {total}, Positive: {positive_count}, Negative: {negative_count}')