from data import TemporalDataset
from tqdm import tqdm
import json

# Load configuration from specified path
with open('config.json') as f:
    config = json.load(f)
    
dataset = TemporalDataset(config, k=3, m_d=50)

dataset.split = 'test'

total = 0
positive_count = 0
negative_count = 0

iteration = 0

steps_per_checkpoint = 10

total_pos_count = 0
checkpoint_pos_count = 0
checkpoint_pos_found = 0

for data in tqdm(dataset):
    # Print position of non zero value
    print(data['central_edge'].nonzero(as_tuple=True))
    total += data['paths'].shape[0]
    positive_count += data['labels'].sum()
    negative_count += data['labels'].shape[0] - data['labels'].sum()
    
    data['labels'] = data['labels'].squeeze(1)
    
    # print(data['masks'][data['labels'] == 1])
    # print(data['paths'][data['labels'] == 1][data['masks'][data['labels'] == 1] == 1])
    
    total_pos_count += data['positives_count']
    checkpoint_pos_count += data['positives_count']
    checkpoint_pos_found += data['positives_found']
        
    iteration += 1
    
    if iteration % steps_per_checkpoint == 0:
        print(f'[Iteration #{iteration}] Total number of samples: {total}, Positive: {positive_count}, Negative: {negative_count}')
        print(f'Hit rate: {checkpoint_pos_found/checkpoint_pos_count:.5f} ({checkpoint_pos_found}/{checkpoint_pos_count}), Cumulative hit rate: {positive_count/total_pos_count:.5f} ({positive_count}/{total_pos_count})\n\n')
        checkpoint_pos_count = checkpoint_pos_found = 0
    
print(f'[Final] Total number of samples: {total}, Positive: {positive_count}, Negative: {negative_count}')
print(f'Cumulative hit rate: {positive_count/total_pos_count:.5f} ({positive_count}/{total_pos_count})\n\n')