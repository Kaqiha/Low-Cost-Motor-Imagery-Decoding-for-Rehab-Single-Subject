# UK BCI: Code Version History

Rest vs move classification from 8-channel EEG (C4, C3, Cz, Fz, PO7, Pz, PO8, Oz) at 250 Hz.

## Versions

| Ver | Preprocessing | Frequency handling | Features / classifier | Evaluation & output | Change vs previous |
|---|---|---|---|---|---|
| **V1** | Concatenate the 3 training files per subject, then filter, then epoch (0.5-4.5 s after cue) | Single band, 7-30 Hz | CSP (4 components) + LDA | CV accuracy on training epochs; predict test epochs | Baseline |
| **V2** | Same as V1 | Filter bank: (7-13) and (18-24) Hz, probabilities averaged across bands | CSP + LDA per band | Mean CV score per band over subjects 1-20, `results.csv` | Replaced one wide band with two targeted bands |
| **V3** | Filter before concatenating the files (see note 1) | **Stage 1 (scan):** 29 one-hertz bands, (i, i+1) for i = 1..29, CV score plotted per band. **Stage 2:** manually pick the best-performing ranges from the plot and use them **for each subject specifically** | Same CSP + LDA per selected band, probabilities averaged | CV score vs band plot guides the manual choice; predictions written to `result.csv` | Replaced guessed bands with a data-driven scan plus per-subject band selection |

**Shared across versions:** 250 Hz, 8 channels, CSP with Ledoit-Wolf regularisation, `ShuffleSplit` 80/20 with seed 67.

## Notes on the current code

1. **Filtering order does not match V3 as described.** `convertRaw` concatenates the CSVs first and then filters, so filter-before-concatenation is not implemented in the uploaded files. Filtering after concatenation can leak edge transients across run boundaries.
2. **Stage 2 is not in the uploaded files.** The notebook only contains the scan (subject 1, all 29 bands, probabilities averaged over all of them). The per-subject selected ranges are not in the code yet.
3. **V2 band mismatch.** The `.py` uses `(8,13)`, not `(7,13)`.
4. **File mapping.** The `.py` is V2 (subjects 1-20, 2 bands). The notebook is the V3 scan stage.
5. **Selection bias.** Choosing bands because they scored best in CV, then reporting that same CV score, makes the estimate optimistic. This matters most with per-subject choices, since each subject has few epochs.
6. **Output format.** The notebook writes `result.csv` with `index=True` (extra unnamed column); the `.py` uses `index=False`.
