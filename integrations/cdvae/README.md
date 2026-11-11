# CDVAE 接口

`evaluate.py` 配合完整 CDVAE 源码树使用，依赖 `scripts/eval_utils.py` 和 `cdvae` 包。
将该文件配置到对应 CDVAE 工作副本的 `scripts/evaluate.py`，并按训练检查点设置模型路径和采样参数。

该接口包含 reconstruction、generation 和 optimization。本文使用 generation 生成结构，不使用 NLO 标签进行潜空间性质优化。
模型检查点、hparams.yaml 及缩放器从 [本文 Figshare](https://doi.org/10.6084/m9.figshare.31849795.v2) 获取。
