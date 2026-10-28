# Low-Cost-Motor-Imagery-Decoding-for-Rehab-Single-Subject-
Code for the Low Cost Motor Imagery Decoding for Rehab (Single Subject) competition hosted by UK BCI Consortium

# UK BCI: Code Version History

Rest vs move classification from 8-channel EEG (C4, C3, Cz, Fz, PO7, Pz, PO8, Oz) at 250 Hz.

## Versions

| Ver | Preprocessing | Frequency handling | Features / classifier | Evaluation & output | Change vs previous |
|---|---|---|---|---|---|
| **V1** | Concatenate the 3 training files per subject, then filter, then epoch (0.5-4.5 s after cue) | Single band, 7-30 Hz | CSP (4 components) + LDA | CV accuracy on training epochs; predict test epochs | Baseline |
| **V2** | Same as V1 | Filter bank: (7-13) and (18-24) Hz. Each band has its own CSP+LDA and predicted probabilities are averaged | CSP + LDA per band, probability averaging | Mean CV score per band over subjects 1-20, `results.csv` written | Replaced one wide band with two targeted bands (roughly mu and beta) |
| **V3** | Filter before concatenating the files (see note 1) | 29 one-hertz bands, (i, i+1) for i = 1..29, each scored separately | Same CSP + LDA per band, probabilities averaged over all bands | CV score plotted against band to find the best frequency region by eye | Replaced guessed bands with a data-driven scan |

**Shared across versions:** 250 Hz, 8 channels, CSP with Ledoit-Wolf regularisation, `ShuffleSplit` 80/20 with seed 67.

## Notes on the current code

1. **Filtering order does not match V3 as described.** `convertRaw` concatenates the CSVs first and then filters, so filter-before-concatenation is not implemented yet. Filtering after concatenation can leak edge transients across run boundaries.
2. **V2 band mismatch.** The `.py` uses `(8,13)`, not `(7,13)`.
3. **File mapping.** The `.py` is V2 (subjects 1-20, 2 bands). The notebook is V3 (subject 1, 29 bands).
4. **V3 averages all 29 bands equally**, including poorly scoring ones.
5. **Output format.** The notebook writes `result.csv` with `index=True` (extra unnamed column); the `.py` uses `index=False`.

## Candidate next versions

| Ver | Idea |
|---|---|
| V4a | Filter each file before concatenating |
| V4b | Average only the top-scoring bands |
| V4c | Pick bands automatically per subject instead of by eye |
| V4d | Move band selection inside the cross-validation loop to avoid selection bias |
