import torch
import torchvision
from sklearn.model_selection import train_test_split
import json
import pandas as pd
with open('data/annotations/labeled.json', 'r') as f:
    data = json.load(f)
data = pd.DataFrame(data)
train, test = train_test_split(data, test_size=0.1, shuffle=True, stratify=data['label'])

with open('data/annotations/train.json', 'w') as f:
    json.dump(train, f)

# save test data to json file
with open('data/annotations/test.json', 'w') as f:
    json.dump(test, f)