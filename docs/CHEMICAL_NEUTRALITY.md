# Chemical-neutrality screening

Charge-screening rules follow SI Table S3. The configuration file is `src/resources/oxidation_states_si.json`, including all listed alternatives such as Tl +1/+3, Pb +2/+4, S -2/+6, Se and Te -2/+4/+6, and Sb +3/+5.

`generated.neutrality` checks whether an allowed oxidation-state assignment gives exactly zero stoichiometric net charge. It returns:

- neutral: a zero-charge assignment was found and is recorded.
- not_neutral: the allowed oxidation states cannot yield zero net charge.
- review: an element is absent from Table S3, or a mixed-valence assignment needs further examination.
- invalid: the input cannot be parsed as a valid composition.

The input is a UTF-8 CSV with a named formula column:

```bash
python -m generated.neutrality input.csv neutrality.csv --formula-column formula
```

The output preserves input rows and appends the status, reason, and oxidation-state assignment. The output path must be a new file.
The default search assigns one allowed oxidation state to each element. If mixed valence could change the conclusion, the composition is sent for review rather than automatically rejected.
Elements absent from Table S3 require case-by-case assessment, as stated in the SI. An explicitly reviewed extension can be supplied with `--states-file`.
Charge neutrality does not establish structural stability or synthetic accessibility.
