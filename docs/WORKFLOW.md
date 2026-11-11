# Workflow and interfaces

## Data preparation and generation

Paper datasets, models, and calculation results are available from [Figshare](https://doi.org/10.6084/m9.figshare.31849795.v2).
Oxygen-containing datasets are BCN_40, BP_40, CSi_40, NP_40, and SiPS_30. Oxygen-excluded datasets are S_20, PS_20, AsSe_20, SbTe_20, PAsSb_20, and SSeTe_20.
`scripts/data/` provides Materials Project retrieval, element filtering, and dataset splitting. Set the relevant input directories and filtering configuration before use.
`build_cdvae_splits.py` uses `predictor.utils.shuffle` for data shuffling and splitting.
The CDVAE generation interface is under `integrations/cdvae/`. Each model generates 2,000 structures in the paper workflow.

## Generated structures

`generated.structures.pt_to_structures(pt_file)` reads atom types, coordinates, and lattice parameters from the first evaluation dimension of the generated output.
`structures_to_cifs` exports CIF files; `strutures_to_csv` exports basic structure information; `elem_counts_of_pt` and `elem_counts_of_structures` summarize element occurrences.

## Compositional and structural features

`predictor.data` combines 256 features from max/min/avg/wavg statistics of 64 elemental properties with 32 CGCNN embedding features, giving 288 features in total.
Inputs include CIF files, a headerless id_prop.csv, atom_init.json, and the official CGCNN formation-energy checkpoint.
Set the data and pretrained-model paths in the corresponding functions or scripts.

| Interface | CSV layout |
|---|---|
| predictor.data.get_dataset output | id, feature..., target |
| generated.predict.get_dataset_from_csv input | id, feature... |
| predictor.train.get_dataset_from_csv_file input | id, delta_n, dij, feature... |

Match the label layout of Figshare single-property datasets to the interface being called; these layouts are not interchangeable.

## Classifier training and prediction

`predictor.train` provides binary classification, low-variance feature filtering, feature-importance ranking, randomized search, cross-validation, SMOTE, and ensemble training.
Specify LightGBM, task thresholds, selected features, and model count explicitly; generic function defaults do not represent every paper task.
The four classification thresholds are dij=0.39 and 3.9 pm/V, and delta_n=0.04 and 0.08, with 120, 20, 30, and 40 selected features, respectively.
Each task uses 100 models, with at least 80 positive votes required for acceptance.

`load_pre_predictors` reads model files and their feature-name table. `get_soft_predictions` returns the average discrete prediction, which is the positive-vote fraction.
The caller applies the 0.8 acceptance threshold; the function's positive_threshold parameter does not internally threshold its output.
`models` repeatedly shuffles the data and uses 90% of the samples for training.

## General properties and screening criteria

See [model sources](../models/README.md) for ALIGNN downloads. Formation energy uses the JARVIS model jv_formation_energy_peratom_alignn.
Chemical-neutrality criteria follow [SI Table S3](CHEMICAL_NEUTRALITY.md).

| Stage | DUV | MIR |
|---|---|---|
| ML stability screening | Eform < 0; Ehull < 0.2 eV/atom | Same |
| ML bandgap screening | PBE bandgap >= 4.0 eV | PBE bandgap >= 2.0 eV |
| ML NLO screening | dij >= 0.39 pm/V; delta_n >= 0.04 | dij >= 3.9 pm/V; delta_n >= 0.04 |
| Bandgap and birefringence in first-principles validation | PBE bandgap >= 3.5 eV; delta_n >= 0.04 | PBE bandgap >= 1.5 eV; delta_n >= 0.03 |

After ML screening, representative candidates are selected for first-principles calculations based on structural diversity and chemical judgment.

## Analysis and result processing

- analysis.structure_statistics: element counts, atomic-site counts, anionic centers, and metal-element statistics.
- tsne_composition.ipynb: t-SNE of element-fraction and stoichiometry descriptors.
- tsne_coordination.ipynb: t-SNE of simple local coordination descriptors.
- generation_validity_uniqueness.ipynb: SMACT validity, minimum-distance validity, and StructureMatcher uniqueness, separate from SI chemical-neutrality screening.
- phase_diagram.ipynb: formation energy and energy above the convex hull using vasprun XML and Materials Project phase-diagram entries.
- phonon_path.ipynb: phonon high-symmetry path generation.
- extract_nlo_properties.py: parsing nlo.log and optical results under lo_*; adapt the input layout when using the Figshare nlo/ hierarchy.
