# UK BCI: Code Version History

Rest vs move classification from 8-channel EEG (C4, C3, Cz, Fz, PO7, Pz, PO8, Oz) at 250 Hz.

## Versions

| Ver | Description |
|---|---|
| **V1** | Uses CSP+LDA on data filtered through (8,30) |
| **V2** | Replaced single filter with [(7,12),(18,24)] bandpass |
| **V3** | Used manually chosen bandpass from high accuracy frequencies and chosen between <br> LDA+CSP and Riemannian Tangent Space | 
| **V4** | Organized processes into classes for better organization |
