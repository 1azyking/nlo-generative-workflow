# 论文数据

数据归档：https://doi.org/10.6084/m9.figshare.31849795.v2

| 归档目录或文件 | 内容 |
|---|---|
| datasets_CDVAE/oxygen-containing/ | BCN_40、BP_40、CSi_40、NP_40、SiPS_30 五组数据；每组 train.csv、val.csv、test.csv |
| datasets_CDVAE/oxygen-excluded/ | S_20、PS_20、AsSe_20、SbTe_20、PAsSb_20、SSeTe_20 六组数据；每组 train.csv、val.csv、test.csv |
| datasets_LGBM/ | dataset_dij.csv、dataset_dn.csv，包含性质标签及组成/结构特征 |
| generated_structures/ | 11 组 CDVAE 生成结果 *_gen.pt |
| calculations.zip | 候选材料 VASP 输入输出及光学性质结果 |

保持归档中的子目录结构，避免不同数据集的 train.csv 等同名文件相互覆盖。
训练模型获取方式见 [models/README.md](../models/README.md)。
