# UK BCI: Code Version History

Rest vs move classification from 8-channel EEG (C4, C3, Cz, Fz, PO7, Pz, PO8, Oz) at 250 Hz.

## Versions

| Ver | Preprocessing | Frequency handling | Features / classifier | Evaluation & output | Change vs previous |
|---|---|---|---|---|---|
| **V1** | Concatenate the 3 training files per subject, then filter, then epoch (0.5-4.5 s after cue) | Single band, 7-30 Hz | CSP (4 components) + LDA | CV accuracy on training epochs; predict test epochs | Baseline |
| **V2** | Same as V1 | Filter bank: (7-13) and (18-24) Hz, probabilities averaged across bands | CSP + LDA per band | Mean CV score per band over subjects 1-20, `results.csv` | Replaced one wide band with two targeted bands |
| **V3** | Filter before concatenating the files (see note 1) | **Stage 1 (scan):** 29 one-hertz bands, (i, i+1) for i = 1..29, CV score plotted per band. **Stage 2:** manually pick the best-performing ranges from the plot and use them **for each subject specifically** | Same CSP + LDA per selected band, probabilities averaged | CV score vs band plot guides the manual choice; predictions written to `result.csv` | Replaced guessed bands with a data-driven scan plus per-subject band selection |

**Shared across versions:** 250 Hz, 8 channels, CSP with Ledoit-Wolf regularisation, `ShuffleSplit` 80/20 with seed 67.
