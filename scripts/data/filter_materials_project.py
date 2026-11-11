import csv

# original_csv_file = 'C:/Users/13701/Desktop/gm_cifs/cdvae/v32/oxysalts/originalDataset.csv'
original_csv_file = 'C:/Users/13701/Desktop/gm_cifs/cdvae/v32/oxyfrees/originalDataset.csv'
filtered_csv_file = original_csv_file.replace('originalDataset', 'filteredDataset')
filtered2_csv_file = original_csv_file.replace('originalDataset', 'filtered2Dataset')
known_crys_csv_file = 'C:/Users/13701/Desktop/gm_cifs/cdvae/knownNLOcrys.csv'

data = []
with open(original_csv_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        data.append(row)
headrow = data[0]
n_crys_0 = len(data[1:])
ids = [row[0] for row in data[1:]]
info = [row[1:] for row in data[1:]]
id_info_dict = dict(zip(ids, info))
elems_str = [row[-1] for row in data[1:]]
elems = []
for e in elems_str:
    elems.append(eval(e))
id_elem_dict = dict(zip(ids, elems))

def metallic_element_including_check(id_elem_dict: dict):
    non_metallic_elements = ['H', 'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se',
                             'Br', 'Te', 'I', 'At', 'He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn']
    filter_ids, filter_elems = [], []
    for id in id_elem_dict.keys():
        elem = id_elem_dict[id]
        counts = 0
        for e in elem:
            if e not in non_metallic_elements:
                counts += 1
            if counts != 0:
                filter_ids.append(id)
                filter_elems.append(elem)
    return dict(zip(filter_ids, filter_elems))

def element_excluding_check(id_elem_dict: dict, exclude_elems_list=None, exclude_elems_class=None):
    assert exclude_elems_list is not None or exclude_elems_class is not None, 'exclude_elems must not be None'
    if exclude_elems_list is not None:
        assert type(exclude_elems_list) is list
        exclude_elems = exclude_elems_list
    elif exclude_elems_list is None and exclude_elems_class is not None:
        assert exclude_elems_class in ['laacs', 'color', 'nbgas', 'hydrogen'], 'invalid exclude_elems_class'
        if exclude_elems_class == 'laacs':
            exclude_elems = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                             'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
        elif exclude_elems_class == 'color':
            exclude_elems = ['V', 'Cr', 'Mn', 'Tc', 'Re', 'Fe', 'Ru', 'Os', 'Co', 'Rh', 'Ir', 'Ni', 'Pd', 'Pt', 'Cu']
        elif exclude_elems_class == 'nbgas':
            exclude_elems = ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn']
        elif exclude_elems_class == 'hydrogen':
            exclude_elems = ['H']
    filter_ids, filter_elems = [], []
    for id in id_elem_dict.keys():
        elem = id_elem_dict[id]
        counts = 0
        for e in elem:
            if e in exclude_elems:
                counts += 1
        if counts == 0:
            filter_ids.append(id)
            filter_elems.append(elem)
    return dict(zip(filter_ids, filter_elems))

def element_check(id_elem_dict: dict):
    id_elem_dict_1 = metallic_element_including_check(id_elem_dict)
    id_elem_dict_2 = element_excluding_check(id_elem_dict_1, exclude_elems_class='laacs')
    id_elem_dict_3 = element_excluding_check(id_elem_dict_2, exclude_elems_class='color')
    id_elem_dict_4 = element_excluding_check(id_elem_dict_3, exclude_elems_class='nbgas')
    id_elem_dict_5 = element_excluding_check(id_elem_dict_4, exclude_elems_class='hydrogen')
    return id_elem_dict_5

filtered_elem_dict = element_check(id_elem_dict)

known_ids = []
with open(known_crys_csv_file, 'r') as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        known_ids.append(row[0])


filtered_data = []
for i in filtered_elem_dict.keys():
    if i not in known_ids:
        datarow = []
        datarow.append(i)
        info = id_info_dict[i]
        for j in info:
            datarow.append(j)
        filtered_data.append(datarow)
filtered_data.insert(0, headrow)

with open(filtered2_csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(filtered_data)
