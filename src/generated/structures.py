import os

import csv

import torch

from pymatgen.core.structure import Structure, Lattice

from pymatgen.io.cif import CifWriter

from collections import Counter

import periodictable

def pt_to_structures(pt_file):
    """
    Convert CDVAE generated/reconstructed results (.pt file) to pymatgen.Structure objects
    Args:
        pt_file(str): file of CDVAE generated/reconstructed results (.pt file)
    Returns:
        a list of pymatgen.Structure objects
    """
    assert os.path.exists(pt_file), "path does not exist: {}".format(pt_file)
    gen_mats = torch.load(pt_file)
    contents = ["eval_setting", "frac_coords", "num_atoms", "atom_types", "lengths", "angles",
                "all_frac_coords_stack", "all_atom_types_stack", "time"]
    coords = gen_mats[contents[1]].numpy()[0]
    n_atoms = gen_mats[contents[2]].numpy()[0]
    atom_types = gen_mats[contents[3]].numpy()[0]
    lengths, angles = gen_mats[contents[4]].numpy()[0], gen_mats[contents[5]].numpy()[0]
    n_crys = len(n_atoms)
    print(f"{n_crys} crystals are generated")
    coords_, atom_types_ = [], []
    previous_n_atoms = 0
    for i in range(n_crys):
        _coords_ = coords[previous_n_atoms:previous_n_atoms + n_atoms[i]]
        coords_.append(_coords_)
        _atom_types_ = atom_types[previous_n_atoms:previous_n_atoms + n_atoms[i]]
        atom_types_.append(_atom_types_)
        previous_n_atoms += n_atoms[i]
    coords, atom_types = coords_, atom_types_
    structures = []
    for i in range(n_crys):
        _lengths_, _angles_ = lengths[i], angles[i]
        _lattice_ = Lattice.from_parameters(a=_lengths_[0], b=_lengths_[1], c=_lengths_[2],
                                            alpha=_angles_[0], beta=_angles_[1], gamma=_angles_[2])
        _coords_ = coords[i]
        _species_ = atom_types[i]
        _structure_ = Structure(lattice=_lattice_, species=_species_, coords=_coords_)
        structures.append(_structure_)
    return structures

def structures_to_cifs(structures: list, cifs_path, pt_file_name=None):
    """
    Save structures to cif files
    Args:
        structures(list): a list of pymatgen.Structure objects
        cifs_path(str): directory to save the cif files
        pt_file_name(str): name of CDVAE generated/reconstructed results
            if not None, cif to save will be named with a prefix of the name of pt file
    Returns:
        No returns; success message
    """
    assert os.path.exists(cifs_path), "path does not exist: {}".format(cifs_path)
    n_cifs = 0
    for s in range(len(structures)):
        structure = structures[s]
        assert type(structure) is Structure, "structures should be a list of pymatgen.Structure objects"
        if pt_file_name is None:
            cif_name = "structure_" + str(s+1) + ".cif"
        else:
            cif_name = pt_file_name + "_structure_" + str(s+1) + ".cif"
        cif_path = os.path.join(cifs_path, cif_name)
        cif_writer = CifWriter(structure, write_magmoms=False)
        cif_writer.write_file(cif_path)
        n_cifs += 1
    print(f"{n_cifs} cif files are saved")

def strutures_to_csv(structures: list, csv_file, pt_file_name=None):
    """
    Save basic information of structures to a csv file
    including id(structure_i), formula, num_sites and num_elements
    Args:
        structures(list): a list of pymatgen.Structure objects
        csv_file(str): file to save basic information of structures
        pt_file_name(str): name of CDVAE generated/reconstructed results
            if not None, cif to save will be named with a prefix of the name of pt file
    Returns:
        No returns; success message
    """
    ids, formulas, n_sites, n_elems = [], [], [], []
    error_ids = []
    for s in range(len(structures)):
        structure = structures[s]
        assert type(structure) is Structure, "structures should be a list of pymatgen.Structure objects"
        if pt_file_name is None:
            id_ = "structure_" + str(s+1)
        else:
            id_ = pt_file_name + "_structure_" + str(s + 1)
        try:
            ids.append(id_)
            formulas.append(structure.composition.reduced_formula)
            n_sites.append(structure.num_sites)
            n_elems.append(len(structure.elements))
        except:
            error_ids.append(ids)
    with open(csv_file, "w", newline="") as file:
        csv_writer = csv.writer(file)
        for i in range(len(ids)):
            datarow = []
            datarow.append(ids[i])
            datarow.append(formulas[i])
            datarow.append(n_sites[i])
            datarow.append(n_elems[i])
            csv_writer.writerow(datarow)
    if len(error_ids) > 0:
        print(f"{len(error_ids)} errors occurred in {error_ids}")
    print(f"info of {len(ids)} structures are saved to {csv_file}")

def list_to_count_dict(lst: list, switch_atom_types_to_chemical_element=True):
    """
    Converts a list to a dictionary that counts the elements and their counts of occurrences
    Args:
        lst(list):
    Returns:
        a dictionary(keys: elements; values: counts of corresponding occurrences)
    """
    element_counts = Counter(lst)
    sorted_elements = sorted(element_counts.items(), key=lambda x: x[1], reverse=True)
    keys, values = [], []
    for i in range(len(sorted_elements)):
        if switch_atom_types_to_chemical_element:
            keys.append(periodictable.elements[sorted_elements[i][0]].symbol)
        else:
            keys.append(sorted_elements[i][0])
        values.append(sorted_elements[i][1])
    element_counts_dict = dict(zip(keys, values))
    return element_counts_dict

def dicts_to_list(dicts: list):
    """
    Converts a list of dictionaries with the same keys to a list
    Args:
        dicts(list): a list of dictionaries with the same keys
    Returns:
        a list([[key1, value1-1, value2-1...], [key2, value1-2, value2-2...]...])
    """
    for dct in dicts:
        assert type(dct) is dict, "input must be a list of dictionaries"
    dct1 = dicts[0]
    lst = []
    for key in dct1.keys():
        _lst_ = []
        _lst_.append(key)
        for dct in dicts:
            try:
                _lst_.append(dct[key])
            except KeyError:
                print("all the input dicts must have the same keys")
                break
        lst.append(_lst_)
    return lst

def elem_counts_of_pt(pt_file: str):
    """
    Count the occurrences of elements in CDVAE generated/reconstructed results (.pt file)
    Args:
        pt_file(str): file of CDVAE generated/reconstructed results (.pt file)
    Returns:
        elem_counts(list): [[elem, occur_count_site, occur_count_crystal]*n_elem]
    """
    assert os.path.exists(pt_file), "path does not exist: {}".format(pt_file)
    gen_mats = torch.load(pt_file)
    n_atoms = gen_mats["num_atoms"].numpy()[0]
    atom_types = gen_mats["atom_types"].numpy()[0]  # to get occurrence counts in sites
    n_crys = len(n_atoms)
    previous_n_atoms, atom_types_, atom_types_cry = 0, [], []
    for i in range(n_crys):
        _atom_types_ = atom_types[previous_n_atoms:previous_n_atoms + n_atoms[i]]
        atom_types_.append(_atom_types_)
        previous_n_atoms += n_atoms[i]
    for i in atom_types_:
        set_i = list(set(i))
        for j in set_i:
            atom_types_cry.append(j)  # to get occurrence counts in crystals
    elem_counts_sites = list_to_count_dict(atom_types)
    elem_counts_crys = list_to_count_dict(atom_types_cry)
    elem_counts = dicts_to_list([elem_counts_sites, elem_counts_crys])
    head = ["element", "site_occurrence_counts", "crystal_occurrence_counts"]
    elem_counts.insert(0, head)
    return elem_counts

def elem_counts_of_structures(structures: list):
    """
    Count the occurrences of elements in structures
    Args:
        structures(list): list of pymatgen.Structure objects
    Returns:
        elem_counts(list): [[elem, occur_count_site, occur_count_crystal]*n_elem]
    """
    elems_crys, elems_sites = [], []
    for s in range(len(structures)):
        structure = structures[s]
        assert type(structure) is Structure, "structures should be a list of pymatgen.Structure objects"
        elems_crys.extend([str(elem.symbol) for elem in structure.elements])
        elems_sites.extend([str(site.specie.symbol) for site in structure.sites])
    elem_counts_sites = list_to_count_dict(elems_sites, switch_atom_types_to_chemical_element=False)
    elem_counts_crys = list_to_count_dict(elems_crys, switch_atom_types_to_chemical_element=False)
    elem_counts = dicts_to_list([elem_counts_sites, elem_counts_crys])
    head = ["element", "site_occurrence_counts", "crystal_occurrence_counts"]
    elem_counts.insert(0, head)
    return elem_counts

def list_to_csv(lst: list, csv_file):
    """
    Save a list to a csv file
    Args:
        lst(list):
        csv_file(str): file to save the input list
    Returns:
        No returns; success message
    """
    with open(csv_file, "w", newline="") as file:
        csv_writer = csv.writer(file)
        for i in range(len(lst)):
            csv_writer.writerow(lst[i])
    print(f"done to {csv_file}")
