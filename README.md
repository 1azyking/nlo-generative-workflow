# JPCL NLO workflow

本项目提供无机非线性光学晶体发现工作流的配套代码，包含 CDVAE 结构生成接口、组成和结构特征提取、LightGBM 分类、SI 电中性筛选及结果分析。

Yu et al., *J. Phys. Chem. Lett.* **2026**, 17, 6463–6471.
论文：https://doi.org/10.1021/acs.jpclett.6c01026
数据与训练模型：https://doi.org/10.6084/m9.figshare.31849795.v2

## 目录

```text
src/
  CGCNN/           晶体图数据与网络
  predictor/       特征提取、分类器训练及辅助函数
  generated/       生成结构处理、集成预测和电中性筛选
  analysis/        结构统计
  resources/       SI 表 S3 氧化态配置
scripts/data/      Materials Project 数据准备
scripts/postprocess/ 结构导出和光学结果提取
integrations/cdvae/ CDVAE 采样评估接口
notebooks/         组成、结构、稳定性及声子路径分析
assets/cgcnn/      原子初始化特征表
environments/      依赖与环境说明
data/              论文数据获取说明
models/            训练模型和官方预训练权重获取说明
docs/              工作流和电中性规则
licenses/          第三方许可声明
```

## 环境与安装

依赖见 [environments/README.md](environments/README.md)。在相应环境中安装项目代码：

```bash
python -m pip install -e . --no-deps
```

`src/` 下的包可直接导入，例如 `from generated.structures import pt_to_structures`。
使用独立脚本和 notebooks 前，设置其中的数据目录、模型目录和输出目录。
Materials Project API 凭据通过 `MP_API_KEY` 环境变量提供，格式见 `.env.example`。

## 数据和模型

- [论文 Figshare](https://doi.org/10.6084/m9.figshare.31849795.v2)：CDVAE 数据、检查点与缩放器，LightGBM 特征数据与四组分类器，生成结构和第一性原理结果。
- [CGCNN 官方预训练权重](https://github.com/txie-93/cgcnn/tree/master/pre-trained)：使用 formation-energy-per-atom.pth.tar；原子特征表为 assets/cgcnn/atom_init.json。
- [ALIGNN 官方模型接口](https://github.com/usnistgov/alignn/blob/main/alignn/pretrained.py)：形成能使用 JARVIS 的 jv_formation_energy_peratom_alignn。

具体目录和模型链接见 [data/README.md](data/README.md) 与 [models/README.md](models/README.md)。
工作流接口见 [docs/WORKFLOW.md](docs/WORKFLOW.md)，电中性判据见 [docs/CHEMICAL_NEUTRALITY.md](docs/CHEMICAL_NEUTRALITY.md)。

## 引用与第三方许可

论文引用见 `CITATION.cff`。第三方声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 及 `licenses/`。
论文 Figshare 数据采用 CC BY 4.0。
