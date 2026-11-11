# 工作流与接口

## 数据准备与生成

论文数据、模型和计算结果见 [Figshare](https://doi.org/10.6084/m9.figshare.31849795.v2)。
含氧数据集为 BCN_40、BP_40、CSi_40、NP_40、SiPS_30；无氧数据集为 S_20、PS_20、AsSe_20、SbTe_20、PAsSb_20、SSeTe_20。
`scripts/data/` 提供 MP 获取、元素筛选和数据划分工具，运行前设置相应输入目录和筛选配置。
`build_cdvae_splits.py` 使用 `predictor.utils.shuffle` 划分数据。
CDVAE 生成接口位于 `integrations/cdvae/`；论文每个模型生成2000个结构。

## 生成结构处理

`generated.structures.pt_to_structures(pt_file)` 读取生成结果首个评估维度的原子、坐标和晶格。
`structures_to_cifs` 导出 CIF，`strutures_to_csv` 导出结构基本信息，`elem_counts_of_pt` 和 `elem_counts_of_structures` 统计元素分布。

## 组成与结构特征

`predictor.data` 将64种元素属性的 max/min/avg/wavg 共256列特征与32维 CGCNN 嵌入组合为288列特征。
输入需要 CIF、无表头 id_prop.csv、atom_init.json 和官方 CGCNN 形成能检查点。
数据和预训练模型路径在相应函数或脚本中设置。

| 接口 | CSV 布局 |
|---|---|
| predictor.data.get_dataset 输出 | id, feature..., target |
| generated.predict.get_dataset_from_csv 输入 | id, feature... |
| predictor.train.get_dataset_from_csv_file 输入 | id, delta_n, dij, feature... |

Figshare 单性质数据应按调用接口匹配标签列布局，不能直接混用上述格式。

## 分类训练与预测

`predictor.train` 提供二分类、低方差特征筛选、重要性排序、随机搜索、交叉验证、SMOTE 和集成训练函数。
调用时显式选择 LightGBM、任务阈值、特征列表和模型数量；通用函数的默认设置不代表所有论文任务的配置。
论文四项任务的阈值为 dij=0.39、3.9 pm/V，Δn=0.04、0.08，对应所选特征数为120、20、30、40。
每项任务使用100个模型，至少80个正类投票作为接受条件。

`load_pre_predictors` 读取模型及配套特征名表；`get_soft_predictions` 返回离散分类结果的平均值，即投票比例。
调用方以0.8阈值判定是否接受；该函数的 positive_threshold 参数不在内部截断输出。
`models` 的实现为重复打乱并取90%样本训练。

## 通用性质与筛选条件

ALIGNN 模型及下载地址见 [模型说明](../models/README.md)。形成能使用 JARVIS 的 jv_formation_energy_peratom_alignn。
电中性判据见 [SI 表 S3 规则](CHEMICAL_NEUTRALITY.md)。

| 阶段 | DUV | MIR |
|---|---|---|
| ML 稳定性筛选 | Eform < 0，Ehull < 0.2 eV/atom | 同左 |
| ML 带隙筛选 | PBE 带隙 ≥ 4.0 eV | PBE 带隙 ≥ 2.0 eV |
| ML NLO 筛选 | dij ≥ 0.39 pm/V，Δn ≥ 0.04 | dij ≥ 3.9 pm/V，Δn ≥ 0.04 |
| 第一性原理验证的带隙和双折射条件 | PBE 带隙 ≥ 3.5 eV，Δn ≥ 0.04 | PBE 带隙 ≥ 1.5 eV，Δn ≥ 0.03 |

通过 ML 筛选后，根据结构多样性和化学判断选择代表性候选进行第一性原理计算。

## 分析与结果处理

- analysis.structure_statistics：元素数、位点数、阴离子中心与金属元素统计。
- tsne_composition.ipynb：元素分数和化学计量特征的 t-SNE。
- tsne_coordination.ipynb：简单局域配位特征的 t-SNE。
- generation_validity_uniqueness.ipynb：SMACT、最短距离有效性和 StructureMatcher 唯一性分析；与 SI 电中性检查分别使用。
- phase_diagram.ipynb：结合 vasprun XML 和 MP 相图条目计算形成能与凸包能。
- phonon_path.ipynb：生成声子高对称路径。
- extract_nlo_properties.py：解析 nlo.log 和 lo_* 下的光学结果；使用 Figshare nlo/ 布局时需对应调整输入布局。
