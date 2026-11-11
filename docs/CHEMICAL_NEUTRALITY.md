# 电中性筛选

电中性判据采用论文 SI 表 S3。配置文件为 `src/resources/oxidation_states_si.json`，包括 Tl +1/+3、Pb +2/+4、S -2/+6、Se和Te -2/+4/+6、Sb +3/+5 等全部允许价态。

`generated.neutrality` 检查允许的氧化态组合是否能使化学计量净电荷严格为零，输出：

- neutral：找到零电荷组合，并记录该组合。
- not_neutral：允许价态范围内无法满足电中性。
- review：含表 S3 未列元素，或需要进一步确定混合价态分配。
- invalid：无法解析为有效化学组成。

输入为带化学式列的 UTF-8 CSV：

```bash
python -m generated.neutrality input.csv neutrality.csv --formula-column formula
```

输出保留原始行并追加判定、理由及价态组合，输出路径须为新文件。
默认规则寻找每种元素采用一个允许价态的组合；若混合价态可能改变结论，则交由人工复核，不自动排除。
表 S3 未列元素按 SI 逐例判断，可通过 `--states-file` 提供经过确认的扩展规则。
电中性判定不代替结构稳定性或合成可行性判断。
