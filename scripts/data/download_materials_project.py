import os

from mp_api.client import MPRester
mpr = MPRester(os.environ["MP_API_KEY"])

from pymatgen.io.cif import CifWriter

# included_elements = ['Al', 'As', 'B', 'Be', 'C', 'I', 'Mo', 'N', 'Nb', 'P', 'S', 'Se', 'Si', 'W']
included_elements = ['P', 'As', 'Te', 'S', 'Se', 'Sb']
for e in included_elements:
    # parent_path = 'C:/Users/13701/Desktop/gm_cifs/cdvae/v32/oxysalts/' + e
    parent_path = 'C:/Users/13701/Desktop/gm_cifs/cdvae/v32/oxyfrees/' + e
    if not os.path.exists(parent_path):
        os.mkdir(parent_path)
    cif_path = os.path.join(parent_path, 'cifs')
    if not os.path.exists(cif_path):
        os.mkdir(cif_path)
    csv_file = os.path.join(parent_path, 'dataset.csv')
    with open(csv_file, 'a') as f:
        f.write(f"mp_id,formula,spg,n_elems,n_sites,e_form,e_above_hull,bandgap,elems\n")

    docs = mpr.materials.summary.search(
        # elements=[e, 'O'],
        # exclude_elements=['H'],
        # num_elements=(3, 7),
        # num_sites=(0, 50),
        # energy_above_hull=(0, 0.08),
        # band_gap=(1, 10),
        # formation_energy=(-10, 2),
        elements=[e],
        exclude_elements=['H', 'O'],
        num_elements=(2, 4),
        num_sites=(0, 50),
        # energy_above_hull=(0, 0.08),
        # band_gap=(1, 10),
        # formation_energy=(-10, 2),
    )
    crys_count, error_ids = 0, []
    for doc in docs:
        mp_id = doc.material_id
        try:
            structure = mpr.get_structure_by_material_id(mp_id)
            cif_writer = CifWriter(structure)
            cif_file = os.path.join(cif_path, '{}.cif'.format(mp_id))
            cif_writer.write_file(cif_file)
            n_elems = len(doc.elements)
            n_sites = int(structure.num_sites)
            formula = str(doc.composition.reduced_formula)
            spg = int(structure.get_space_group_info()[1])
            e_form = round(float(doc.formation_energy_per_atom), 4)
            e_above_hull = round(float(doc.energy_above_hull), 4)
            bandgap = round(float(doc.band_gap), 4)
            elems = [str(e) for e in doc.elements]
            with open(csv_file, 'a') as f:
                f.write(f"{mp_id},{formula},{spg},{n_elems},{n_sites},{e_form},{e_above_hull},{bandgap},\"{elems}\"\n")
            crys_count += 1
        except:
            error_ids.append(mp_id)
    # print(f"oxysalts with {e}:")
    print(f"oxyfrees with {e}:")
    print(f"{crys_count} crystals are downloaded")
    if len(error_ids) > 0:
        print(f"{len(error_ids)} errors: {error_ids}")
    print()
