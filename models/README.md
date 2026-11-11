# Models

## Models trained for the paper

Archive: https://doi.org/10.6084/m9.figshare.31849795.v2

- models_CDVAE/: checkpoints, hparams.yaml, lattice_scaler.pt, and prop_scaler.pt for 11 datasets.
- models_LGBM/: dij_039.zip, dij_390.zip, dn_004.zip, and dn_008.zip.
- Each classifier archive contains predictor_0.pkl through predictor_99.pkl and selected_feature_names.csv. Select and order input features using the companion feature-name table.

## CGCNN

Use the official formation-energy model:
[formation-energy-per-atom.pth.tar](https://raw.githubusercontent.com/txie-93/cgcnn/master/pre-trained/formation-energy-per-atom.pth.tar).

The companion atomic feature table is `assets/cgcnn/atom_init.json`, corresponding to the [official CGCNN atom_init.json](https://raw.githubusercontent.com/txie-93/cgcnn/master/data/sample-regression/atom_init.json).
It contains 92-dimensional initialization vectors for 100 elements.

## ALIGNN

| Property | Official model | Download |
|---|---|---|
| Formation energy | jv_formation_energy_peratom_alignn (JARVIS) | [Model ZIP](https://figshare.com/ndownloader/files/31458679) |
| Energy above the convex hull | jv_ehull_alignn | [Model ZIP](https://figshare.com/ndownloader/files/31458658) |
| PBE bandgap | mp_gappbe_alignn | [Model ZIP](https://figshare.com/ndownloader/files/31458814) |

See the official model registry in [ALIGNN pretrained.py](https://github.com/usnistgov/alignn/blob/main/alignn/pretrained.py).
The ALIGNN interface reads a model ZIP containing config.json and a checkpoint. The historical name mp_gappbe_alignnn in the code refers to the official mp_gappbe_alignn model; keep filenames and configured paths consistent.
