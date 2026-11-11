# 模型

## 本文训练模型

归档：https://doi.org/10.6084/m9.figshare.31849795.v2

- models_CDVAE/：11 个数据集对应的检查点、hparams.yaml、lattice_scaler.pt 和 prop_scaler.pt。
- models_LGBM/：dij_039.zip、dij_390.zip、dn_004.zip、dn_008.zip。
- 每个分类器 ZIP 包含 predictor_0.pkl 至 predictor_99.pkl，以及 selected_feature_names.csv。预测时按配套特征名表选择并排列输入特征。

## CGCNN

使用官方形成能模型：
[formation-energy-per-atom.pth.tar](https://raw.githubusercontent.com/txie-93/cgcnn/master/pre-trained/formation-energy-per-atom.pth.tar)。

配套原子特征表为 `assets/cgcnn/atom_init.json`，对应 [CGCNN 官方 atom_init.json](https://raw.githubusercontent.com/txie-93/cgcnn/master/data/sample-regression/atom_init.json)。
它包含100种元素的92维原子初始化向量。

## ALIGNN

| 性质 | 官方模型 | 下载 |
|---|---|---|
| 形成能 | jv_formation_energy_peratom_alignn（JARVIS） | [模型 ZIP](https://figshare.com/ndownloader/files/31458679) |
| 凸包能 | jv_ehull_alignn | [模型 ZIP](https://figshare.com/ndownloader/files/31458658) |
| PBE 带隙 | mp_gappbe_alignn | [模型 ZIP](https://figshare.com/ndownloader/files/31458814) |

官方模型目录见 [ALIGNN pretrained.py](https://github.com/usnistgov/alignn/blob/main/alignn/pretrained.py)。
ALIGNN 调用函数接收包含 config.json 和 checkpoint 的模型 ZIP。代码中历史名称 mp_gappbe_alignnn 对应官方 mp_gappbe_alignn，配置文件名和路径时应保持对应。
