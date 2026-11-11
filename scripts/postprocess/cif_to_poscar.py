import os
from pymatgen.core import Structure
import csv
from pymatgen.io.vasp import Poscar
from pymatgen.io.cif import CifWriter

cif_filenames = []
csv_file = "C:/Users/13701/Desktop/ttmp.csv"
with open(csv_file) as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    for row in csv_reader:
        cif_filenames.append(row[0])

cif_path = "D:/PycharmProjects/cdvae-local/pt_files/v32/"
# to_path = "D:/PycharmProjects/cdvae-local/pt_files/v32/cand_2411/duv/neutral"

# cif_path = "C:/Users/13701/Desktop/gm_cifs/cdvae/v32/to_plot/5sets_gens"
to_path = "C:/Users/13701/Desktop/tmp_oxyf"

for filename in cif_filenames:
    dirname = filename.split("_structure")[0]
    cif_dir = os.path.join(cif_path, dirname, "cifs")
    # cif_dir = cif_path
    cif_file = os.path.join(cif_dir, f"{filename}.cif")
    structure = Structure.from_file(cif_file)
    structure.to_primitive()
    redu_fml = structure.composition.reduced_formula
    dirname_n = filename.replace("_gen_structure", "")
    to_dir = os.path.join(to_path, f"{dirname_n}_{redu_fml}")
    os.makedirs(to_dir, exist_ok=True)
    cif_writer = CifWriter(structure)
    structure_file = os.path.join(to_dir, f"{redu_fml}.cif")
    cif_writer.write_file(structure_file)
    poscar_file = os.path.join(to_dir, "POSCAR")
    poscar = Poscar(structure)
    poscar.write_file(poscar_file)
    print(f"{dirname_n} done")
