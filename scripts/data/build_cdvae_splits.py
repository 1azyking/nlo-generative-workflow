import csv
import os

parent_path = 'C:/Users/13701/Desktop/gm_cifs/cdvae/v32/oxysalts/'
# cif_pool_path = os.path.join(parent_path, 'elembyelem')
cif_pool_path = os.path.join(parent_path, 'filteredCifs')
dataset_name = 'oxy_x_50'
dataset_path = os.path.join(parent_path, 'datasets/' + dataset_name)
dataset_csv_file = os.path.join(dataset_path, dataset_name + '.csv')
output_csv_file = os.path.join(dataset_path, 'output.csv')

data = []
with open(dataset_csv_file, 'r') as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        data.append(row)

mp_ids, e_forms, e_ahs, bgs = [], [], [], []
for i in range(1, len(data)):
    mp_ids.append(data[i][0])
    e_forms.append(data[i][5])
    e_ahs.append(data[i][6])
    bgs.append(data[i][7])

def get_filelist(parent_path):
    filelist = []
    for home, dirs, files in os.walk(parent_path):
        for filename in files:
            if filename.endswith('.cif'):
                filelist.append(os.path.join(home, filename))
    return filelist

cif_filelist = get_filelist(cif_pool_path)

with open(output_csv_file, 'w', newline='') as f:
    fieldnames = ['mp_id', 'cif', 'e_form', 'e_ah', 'bg']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(mp_ids)):
        cif_filename = mp_ids[i] + '.cif'
        cif_file = [i for i in cif_filelist if i.endswith(cif_filename)][0]
        with open(cif_file, 'r') as cif_f:
            cif_contents = cif_f.read()
        writer.writerow({'mp_id': mp_ids[i], 'cif': cif_contents, 'e_form': e_forms[i], 'e_ah': e_ahs[i], 'bg': bgs[i]})

import numpy as np
data = []
with open(output_csv_file, mode='r', ) as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        data.append(row)
head = data[0]
data = data[1:]
n_data = len(data)
test_ratio, val_ratio = 0.05, 0.05
n_test = int(n_data * test_ratio)
n_val = n_test
n_train = n_data - n_val - n_test
# print(data[0])
from predictor.utils import shuffle
seed1, seed2, seed3 = 2024, 4, 29
data = shuffle(data, seed1)
data = shuffle(data, seed2)
data = shuffle(data, seed3)
headrow = np.array(['', 'material_id','cif','e_form','e_ah','bg'])

def split(dataset, test_ratio=0.1, val_ratio=0.1):
    n_all = len(dataset)
    n_test = int(n_all * test_ratio)
    n_val = int(n_all * val_ratio)
    n_train = n_all - n_test - n_val
    train = dataset[:n_train]
    test = dataset[n_train:n_train+n_test]
    val = dataset[n_train+n_test:]
    return (train, test, val)

def save_csv(dataset, type, path, headrow=None):
    assert type in ['train', 'test', 'val'], 'type must be in ["train", "test", "val"]'
    n_samples = len(dataset)
    dataset = np.array(dataset)
    indices = np.array([i for i in range(n_samples)]).reshape(-1,1)
    data_ = np.concatenate((indices, dataset), axis=1)
    if headrow is not None:
        headrow = np.array(headrow).reshape(1,-1)
        data_ = np.concatenate((headrow, data_), axis=0)
    csv_path = os.path.join(path, type + '.csv')
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for row in data_:
            writer.writerow(row)
    print(type, 'is done')

train, test, val = split(data)
save_csv(train, 'train', dataset_path, headrow)
save_csv(test, 'test', dataset_path, headrow)
save_csv(val, 'val', dataset_path, headrow)
