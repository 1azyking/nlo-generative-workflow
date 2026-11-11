import csv

import os

import torch

import numpy as np

from matminer.featurizers.composition.element import MagpieData

from pymatgen.core.structure import Structure

from CGCNN.cgcnn_model import CrystalGraphConvNet

from CGCNN.cgcnn_data import CIFData, collate_pool

def get_magpie_feature_names(remove_magpie_features=None):
    """
    Returns a list of feature names for Magpie data
    Args:
        remove_magpie_features(list or str): List of/a single feature names to remove from Magpie data
    Returns:
        magpie_props(list): List of Magpie properties to generate features
    """
    magpie_props = ['AllenElectronegativity', 'AtomicRadius', 'AtomicVolume', 'AtomicWeight', 'BoilingT',
                    'BulkModulus', 'Column', 'CovalentRadius', 'Density', 'DipolePolarizability',
                    'ElectronAffinity', 'Electronegativity', 'FirstIonizationEnergy', 'FusionEnthalpy', 'GSbandgap',
                    'GSenergy_pa', 'GSestBCClatcnt', 'GSestFCClatcnt', 'GSmagmom', 'GSvolume_pa',
                    'HHIp', 'HHIr', 'HeatCapacityMass', 'HeatCapacityMolar', 'HeatFusion',
                    'HeatVaporization', 'ICSDVolume', 'IonizationEnergies', 'IsAlkali', 'IsDBlock',
                    'IsFBlock', 'IsMetal', 'IsMetalloid', 'IsNonmetal', 'LogThermalConductivity',
                    'MeltingT', 'MendeleevNumber', 'MiracleRadius', 'MolarVolume', 'NUnfilled',
                    'NValence', 'NdUnfilled', 'NdValence', 'NfUnfilled', 'NfValence',
                    'NpUnfilled', 'NpValence', 'NsUnfilled', 'NsValence', 'Number',
                    'OxidationStates', 'Polarizability', 'Row', 'SecondIonizationEnergy', 'ShearModulus',
                    'SpaceGroupNumber', 'ThermalConductivity', 'VdWRadius', 'ZungerPP-r_d', 'ZungerPP-r_p',
                    'ZungerPP-r_pi', 'ZungerPP-r_s', 'ZungerPP-r_sigma', 'n_ws^third', 'phi']
    magpie_props.remove('OxidationStates')
    if remove_magpie_features is not None:
        if type(remove_magpie_features) == list:
            for remove_feature in remove_magpie_features:
                magpie_props.remove(remove_feature)
        elif type(remove_magpie_features) == str:
            magpie_props.remove(remove_magpie_features)
        else:
            print(f'remove_magpie_features type {type(remove_magpie_features)} not supported')
    return magpie_props

def prop_to_path(prop: str, parent_path='C:/Users/13701/Desktop/gm_cifs/cgcnn/mv_dataset/'):
    assert prop in ['dij', 'dn'], 'Invalid property!'
    return os.path.join(parent_path, prop)

def check_prop_path(prop):
    """

    Args:
        prop:

    Returns:

    """
    data_path = prop_to_path(prop)
    assert os.path.exists(data_path), 'Data path does not exist!'
    id_prop_file = os.path.join(data_path, 'id_prop.csv')
    assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
    return (data_path, id_prop_file)

def load_original_structures(prop=None, data_path=None):
    """

    Args:
        prop: 'dn' or 'dij'

    Returns:

    """
    if prop is not None:
        (data_path, id_prop_file) = check_prop_path(prop)
    elif data_path is not None:
        data_path = data_path
        id_prop_file = os.path.join(data_path, 'id_prop.csv')
    else:
        print('prop or data_path must be provided')
    assert os.path.isfile(os.path.join(data_path, 'id_prop.csv')), "id_prop.csv file is not found"
    # (data_path, id_prop_file) = check_prop_path(prop)
    ids, targets, structures = [], [], []
    with open(id_prop_file, 'r', newline='') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            ids.append(row[0])
            targets.append(float(row[-1]))
        for id in ids:
            cif_path = os.path.join(data_path, f"{id}.cif")
            structure = Structure.from_file(cif_path)
            structures.append(structure)
        assert len(ids) == len(targets) == len(structures), \
            'Numbers of ids/targets/structures are not equal!'
    return (ids, structures, targets)

def get_magpie_features(structure, remove_magpie_features=None):
    """

    Args:
        structure:
        magpie_props:

    Returns:

    """
    elems = structure.elements
    elem_fracs = [fraction for element, fraction in structure.composition.items()]
    magpie = MagpieData()
    magpie_feas = []
    magpie_props = get_magpie_feature_names(remove_magpie_features)
    for e in range(len(elems)):
        elem = elems[e]
        magpie_feas_elem = []
        for p in range(len(magpie_props)):
            magpie_feas_elem.append(magpie.get_elemental_property(elem=elem, property_name=magpie_props[p]))
        magpie_feas.append(magpie_feas_elem)
    magpie_feas = np.array(magpie_feas)
    assert magpie_feas.shape == (len(elems), len(magpie_props))
    magpie_feas = np.ma.masked_invalid(magpie_feas)
    _max_ = np.max(magpie_feas, axis=0)
    _min_ = np.min(magpie_feas, axis=0)
    _avg_ = np.mean(magpie_feas, axis=0)
    _wei_avg_ = np.average(magpie_feas, axis=0, weights=elem_fracs)
    magpie_features = np.vstack([_max_, _min_, _avg_, _wei_avg_])
    return magpie_features.reshape(-1, order='F')

def get_magpie_dataset(ids, structures, targets):
    features = []
    for structure in structures:
        feature = get_magpie_features(structure)
        features.append(feature)
    features = np.array(features)
    return (ids, features, targets)

def get_cgcnn_dataset(prop=None, data_path=None, pre_model='formation-energy-per-atom'):
    """

    Args:
        prop:

    Returns:

    """
    if prop is not None:
        (data_path, id_prop_file) = check_prop_path(prop)
    elif data_path is not None:
        data_path = data_path
    else:
        print('prop or data_path must be provided')
    assert os.path.isfile(os.path.join(data_path, 'id_prop.csv')), "id_prop.csv file is not found"
    assert os.path.isfile(os.path.join(data_path, 'atom_init.json')), "atom_init.json file is not found"
    cgcnn_dataset = CIFData(root_dir=data_path)
    cgcnn_dataset_list = [cgcnn_dataset[i] for i in range(len(cgcnn_dataset))]
    collate_dataset_list = collate_pool(cgcnn_dataset_list)
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx, crystal_atom_idx = collate_dataset_list[0]
    batch_target, batch_cif_ids = collate_dataset_list[1], collate_dataset_list[2]
    cgcnn_model = CrystalGraphConvNet(orig_atom_fea_len=92, nbr_fea_len=41,
                                      atom_fea_len=64, n_conv=4, h_fea_len=32,
                                      n_h=1, classification=False)
    pre_model_path = 'D:/PycharmProjects/cgcnn/cgcnn-master/pre-trained/' + pre_model + '.pth.tar'
    ckpt = torch.load(pre_model_path)
    state_dict = ckpt['state_dict']
    cgcnn_model.load_state_dict(state_dict)
    cgcnn_input = (batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx, crystal_atom_idx)
    cgcnn_output = cgcnn_model(*cgcnn_input)
    features = cgcnn_output.detach().numpy()
    targets = np.squeeze(batch_target.numpy())
    ids = np.array(batch_cif_ids)
    return (ids, features, targets)

def get_dataset(
        magpie_dataset,
        cgcnn_dataset,
        save_csv=True,
        save_path=None,
):
    magpie_ids, magpie_features, magpie_targets = magpie_dataset
    cgcnn_ids, cgcnn_features, cgcnn_targets = cgcnn_dataset
    cgcnn_ids = list(cgcnn_ids)
    # get mixed dataset
    mixed_features = []
    for i in range(len(magpie_ids)):
        _id_ = magpie_ids[i]
        cgcnn_idx = cgcnn_ids.index(_id_)
        magpie_feature = magpie_features[i]
        cgcnn_feature = cgcnn_features[cgcnn_idx]
        mixed_feature = np.concatenate((magpie_feature, cgcnn_feature))
        mixed_features.append(mixed_feature)
    mixed_features = np.array(mixed_features)
    # get feature names
    statistic_methods = ['max', 'min', 'avg', 'wavg']
    magpie_feature_names, cgcnn_feature_names = [], []
    magpie_props = get_magpie_feature_names()
    for i in magpie_props:
        for j in statistic_methods:
            magpie_feature_names.append(j + i)
    for i in range(cgcnn_features.shape[1]):
        cgcnn_feature_names.append('cgcnn' + str(i))
    mixed_feature_names = magpie_feature_names + cgcnn_feature_names
    # save dataset as csv file
    ids, targets = magpie_ids, magpie_targets
    feature_names, features = mixed_feature_names, mixed_features
    data = []
    for id_, target, feature_row in zip(ids, targets, features):
        data_row = [id_] + list(feature_row) + [target]
        data.append(data_row)
    if save_csv:
        assert save_path is not None, "save path should be provided"
        with open(save_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['id'] + feature_names + ['target'])
            writer.writerows(data)
        print('Data preparation is done and csv file is saved.')
    if not save_csv:
        return (ids, targets, features, feature_names)
