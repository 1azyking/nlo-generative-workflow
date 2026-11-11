# 环境

## 特征提取与分类

依赖列表：

- requirements-ml.in：特征、分类器和预训练模型接口。
- requirements-analysis.in：统计、可视化、Materials Project 查询及晶体分析工具。

```bash
python -m pip install -r environments/requirements-ml.in
python -m pip install -r environments/requirements-analysis.in
python -m pip install -e . --no-deps
```

ML 依赖列表不锁定版本。ALIGNN、DGL、JARVIS、PyTorch、matminer 和 pymatgen 应使用相互兼容的版本；读取既有分类器时，还需与其 LightGBM/scikit-learn 序列化格式兼容。

## CDVAE

CDVAE 使用独立环境及其完整源码树。可参考 cdvaecpu 的主要版本：

| 组件 | 参考版本 |
|---|---|
| Python | 3.9.18 |
| PyTorch | 1.8.1 |
| PyTorch Lightning | 1.3.8 |
| torch-geometric | 2.0.1 |
| hydra-core | 1.1.0 |
| omegaconf | 2.1.2 |

采样接口依赖 CDVAE 的 scripts/eval_utils.py、模型配置及 GemNet 配套文件，使用方式见 [integrations/cdvae/README.md](../integrations/cdvae/README.md)。
