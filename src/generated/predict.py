import joblib

import csv

import os

import shutil

import zipfile

import tempfile

import json

import time

import gc

import torch

import numpy as np

from collections import Counter

from pymatgen.core.structure import Structure

from matminer.featurizers.composition.element import MagpieData

from alignn.models.alignn import ALIGNN, ALIGNNConfig

from jarvis.core.atoms import Atoms

from jarvis.core.graphs import Graph

from predictor.data import load_original_structures, get_magpie_dataset, get_dataset, get_cgcnn_dataset

from CGCNN.cgcnn_data import CIFData, collate_pool

from CGCNN.cgcnn_model import CrystalGraphConvNet

def load_pre_predictors(models_path: str):
    """

    Args:
        models_path:

    Returns:

    """
    assert os.path.exists(models_path), "path does not exist: {}".format(models_path)
    models, selected_feature_names = [], []
    for filename in os.listdir(models_path):
        if filename.endswith('.pkl'):
            models.append(joblib.load(os.path.join(models_path, filename)))
        if filename.endswith('selected_feature_names.csv'):
            feat_name_file = os.path.join(models_path, filename)
            with open(feat_name_file, 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    selected_feature_names.append(row[0])
    return models, selected_feature_names

def make_cgcnn_input(
        cifs_path: str,
):
    """
    Make formal input for CGCNN operation&prediction
    Args:
        cifs_path(str): directory of cif files
    Returns:
        checked_cifs_path(str): same to cifs_path; ready for CGCNN
    """
    if not os.path.exists(os.path.join(cifs_path, 'id_prop.csv')):
        with open(os.path.join(cifs_path, 'id_prop.csv'), 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            for filename in os.listdir(cifs_path):
                if filename.endswith('.cif'):
                    id_prop_row = [filename[:-4], -1]
                    writer.writerow(id_prop_row)
        print("id_prop.csv file is created")
    if not os.path.exists(os.path.join(cifs_path, 'atom_init.json')):
        shutil.copy('D:/PycharmProjects/cdvae-local/pt_files/atom_init.json', cifs_path)
        print("atom_init.json is copied here")
    assert os.path.isfile(os.path.join(cifs_path, 'id_prop.csv')), "id_prop.csv file is not found"
    assert os.path.isfile(os.path.join(cifs_path, 'atom_init.json')), "atom_init.json file is not found"
    print("ready to make cgcnn operations")
    checked_cifs_path = cifs_path
    return checked_cifs_path

def get_pretrained_cgcnn_predictions(
        checked_cifs_path: str,
        pre_cgcnn_model_file: str,
        str_cgcnn_prop=None,
):
    """
    Get predictions from pre-trained CGCNN model
    Args:
        checked_cifs_path(str): directory of format CGCNN input including cif files, atom_init.json and id_prop.csv
        pre_cgcnn_model_file(str): file of pre-trained CGCNN model (.pth.tar)
    Returns:
        prop_preds(dict) with keys 'id' and prop(str_cgcnn_prop)
    """
    cgcnn_dataset = CIFData(root_dir=checked_cifs_path)
    cgcnn_dataset_list = [cgcnn_dataset[i] for i in range(len(cgcnn_dataset))]
    ids = [cgcnn_dataset_list[i][-1] for i in range(len(cgcnn_dataset_list))]
    collate_dataset_list = collate_pool(cgcnn_dataset_list)
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx, crystal_atom_idx = collate_dataset_list[0]
    cgcnn_model = CrystalGraphConvNet(orig_atom_fea_len=92, nbr_fea_len=41,
                                      atom_fea_len=64, n_conv=4, h_fea_len=32,
                                      n_h=1, classification=False)
    ckpt = torch.load(pre_cgcnn_model_file)
    state_dict = ckpt['state_dict']
    cgcnn_model.load_state_dict(state_dict)
    cgcnn_input_to_out = (batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx, crystal_atom_idx, False)
    cgcnn_output = cgcnn_model(*cgcnn_input_to_out)
    cgcnn_predictions = np.squeeze(cgcnn_output.detach().numpy())
    cgcnn_predictions = list(cgcnn_predictions)
    ids.insert(0, 'id')
    cgcnn_predictions.insert(0, str_cgcnn_prop)
    return dict(zip(ids, cgcnn_predictions))

def get_pretrained_alignn_predictions(
        cifs_path: str,
        model_name: str = 'jv_formation_energy_peratom_alignn',
        model_path: str = None,
):
    """
    Get predictions from pre-trained ALIGNN model
    Args:
        cifs_path(str): directory of cif files
        model_name(str): (partial) name of pre-trained ALIGNN model
        model_path(str): path to pre-trained ALIGNN model
                         if None, model would be found under 'D:/PycharmProjects/alignn/pre-trained/'
    Returns:
        prop_preds(dict) with keys 'id' and prop(model_name)
    """
    if model_path is None:
        model_path = 'D:/PycharmProjects/alignn/pre-trained/'
    model_file, str_alignn_prop = None, None
    for filename in os.listdir(model_path):
        if model_name in filename:
            model_file = os.path.join(model_path, filename)
            str_alignn_prop = filename[:-4]
            break
    assert str_alignn_prop is not None, f"model of {model_name} not found in {model_path}"
    with zipfile.ZipFile(model_file, 'r') as zp:
        names = zp.namelist()
        chks, cfg = [], []
        for name in names:
            if 'checkpoint' in name and 'pt' in name:
                chks.append(name)
            if 'config.json' in name:
                cfg.append(name)
        config = json.loads(zp.read(cfg[0]))
        data = zp.read(chks[0])
    model = ALIGNN(ALIGNNConfig(**config['model']))
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filename = tmp.name
        tmp.write(data)
        tmp.flush()
    model.load_state_dict(torch.load(filename)['model'])
    os.remove(filename)
    model.to('cpu')
    model.eval()

    ids, gs, lgs, error_ids = [], [], [], []
    for filename in os.listdir(cifs_path):
        if filename.endswith('.cif'):
            cif_file = os.path.join(cifs_path, filename)
            try:
                atoms = Atoms.from_cif(cif_file, use_cif2cell=False)
                g, lg = Graph.atom_dgl_multigraph(atoms)
                ids.append(filename[:-4])
                gs.append(g.to('cpu'))
                lgs.append(lg.to('cpu'))
            except:
                error_ids.append(filename[:-4])
    if len(error_ids) > 0:
        print(f"{len(error_ids)} errors occurred in cif_to_Atoms: {error_ids}")
    alignn_predictions = []
    for i in range(len(ids)):
        alignn_predictions.append(float(model([gs[i], lgs[i]]).detach().cpu().numpy()))
    assert len(alignn_predictions) == len(ids)
    ids.insert(0, 'id')
    alignn_predictions.insert(0, str_alignn_prop)
    for i in range(len(error_ids)):
        ids.append(error_ids[i])
        alignn_predictions.append(9999.999)
    return dict(zip(ids, alignn_predictions))

def get_sub_features_via_names(features, feature_names, selected_feature_names):
    """
    Get subset of selected_feature_names from features of feature_names
    Args:
        features(np.ndarray): features of feature_names (shape: (n_samples, n_features)
        feature_names(list or 0-dim np.ndarray): list of original feature names
        selected_feature_names(list or 0-dim np.ndarray): list of selected feature names
    Returns:
        selected_features(np.ndarray): subset of selected_feature_names
    """
    assert len(features[0]) == len(feature_names)
    selected_features = []
    for feature_name in selected_feature_names:
        feature_idx = list(feature_names).index(feature_name)
        selected_features.append(features[..., feature_idx])
    return np.array(selected_features).T

def checked_cifs_path_to_dataset(checked_cifs_path):
    assert os.path.isfile(os.path.join(checked_cifs_path, 'id_prop.csv')), "id_prop.csv file is not found; "
    assert os.path.isfile(os.path.join(checked_cifs_path, 'atom_init.json')), "atom_init.json file is not found"
    (ids, structures, targets) = load_original_structures(data_path=checked_cifs_path)
    magpie_dataset = get_magpie_dataset(ids, structures, targets)
    cgcnn_dataset = get_cgcnn_dataset(data_path=checked_cifs_path)
    (ids, targets, features, feature_names) = get_dataset(magpie_dataset, cgcnn_dataset, save_csv=False)
    dataset = (ids, targets, features, feature_names)
    return dataset

def get_dataset_from_csv(csv_file: str):
    assert os.path.isfile(csv_file) and csv_file.endswith(".csv"), f"{csv_file} is invalid"
    datarows = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            datarows.append(row)
    feature_names = datarows[0][1:]
    ids = [datarow[0] for datarow in datarows[1:]]
    targets = [-1 for _ in range(len(ids))]
    features = [datarow[1:] for datarow in datarows[1:]]
    targets, features = np.array(targets, dtype=np.float64), np.array(features, dtype=np.float64)
    dataset = (ids, targets, features, feature_names)
    return dataset

def get_soft_predictions(
        dataset: tuple,
        pre_models,
        selected_feature_names,
        str_target_criterion,
        positive_threshold=0.8,
):
    """

    Args:
        dataset:
        pre_models:
        selected_feature_names:
        str_target_criterion:
        positive_threshold:

    Returns:

    """
    (ids, targets, features, feature_names) = dataset
    selected_features = get_sub_features_via_names(features, feature_names, selected_feature_names)
    predictions = []
    for model in pre_models:
        # predictions.append(model.predict(selected_features))
        # # predictions = np.array(predictions)
        # average_predictions.append(np.mean(np.array(predictions), axis=0))
        preds = model.predict(selected_features)
        preds = np.array(preds)
        predictions.append(preds)
    predictions = np.array(predictions)
    average_predictions = np.mean(np.array(predictions), axis=0)
    # print(average_predictions)
    soft_predictions = average_predictions
    # soft_predictions = []
    # binary-cls
    # for i in range(len(average_predictions)):
    #     if average_predictions[i] >= positive_threshold:
    #         soft_predictions.append(1)
    #     elif average_predictions[i] < (1-positive_threshold):
    #         soft_predictions.append(0)
    #     else:
    #         soft_predictions.append(9999.999)
    # multi-cls
    # for prediction in predictions:
    #     if list(prediction):
    #         counter = Counter(prediction)
    #         most_pred, count = counter.most_common(1)[0]
    #         freq = count / len(prediction)
    #         if freq >= positive_threshold:
    #             soft_predictions.append(most_pred)
    #         else:
    #             soft_predictions.append(9999.999)
    #     else:
    #         soft_predictions.append(9999.999)
    ids, soft_predictions = list(ids), list(soft_predictions)
    ids.insert(0, 'id')
    soft_predictions.insert(0, str_target_criterion)
    id_soft_prediction_dict = dict(zip(ids, soft_predictions))
    return id_soft_prediction_dict

def dicts_to_list(dicts: list):
    """
    Convert a list of dictionaries with the same keys to a list
    Args:
        dicts(list): a list of dictionaries with the same keys
    Returns:
        a list([[key1, value1-1, value2-1...], [key2, value1-2, value2-2...]...])
    """
    for dct in dicts:
        assert type(dct) is dict, 'input must be a list of dictionaries'
    dct1 = dicts[0]
    lst = []
    for key in dct1.keys():
        _lst_ = []
        _lst_.append(key)
        for dct in dicts:
            try:
                _lst_.append(dct[key])
            except KeyError:
                print('all the input dicts must have the same keys')
                break
        lst.append(_lst_)
    return lst

def list_to_dict_with_headrow_as_keys(lst: list):
    """
    Convert a list to a dictionary with list[0] as keys and list[1] as values
    Args:
        lst(list): a 2-D list; n_elements in each row are equal
    Returns:
        dict
    """
    keys = lst[0]
    lst = lst[1:]
    values = []
    for i in range(len(keys)):
        value = []
        for j in range(len(lst)):
            value.append(lst[j][i])
        values.append(value)
    return dict(zip(keys, values))

def list_to_csv(lst: list, csv_file):
    """
    Save a list to a csv file
    Args:
        lst(list):
        csv_file(str): file to save the input list
    Returns:
        No returns; success message
    """
    with open(csv_file, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        for i in range(len(lst)):
            csv_writer.writerow(lst[i])
    print(f"done to {csv_file}")

def dict_to_csv(dct: dict, save_csv_file: str):
    with open(save_csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(dct.keys())
        max_length = max(len(lst) for lst in dct.values())
        for i in range(max_length):
            row = [dct[key][i] if i < len(dct[key]) else '' for key in dct.keys()]
            writer.writerow(row)
    print(f"dict saved to {save_csv_file}")
