# Checkpoint 1 write-up

Junjie Cai (jc1206)

## 1. What exists at prediction time

The Challenge 2012 prediction window is the first 48 hours of an ICU stay. At that time we have admission descriptors (`Age`, `Gender`, `Height`, `ICUType`, admission `Weight`) and every time-stamped measurement in `output/set-a_long.csv`. Whether a variable was ordered is also known.

`In-hospital_death`, `Survival`, and `Length_of_stay` are not available as features. SAPS-I and SOFA in `Outcomes-a.txt` are for description and later scoring, not inputs to leak into a model.

## 2. Table 1 and outcome summary

`output/table1.csv` is grouped by in-hospital death. n = 4000; 554 died (13.85%). Deaths are older (mean 70.0 vs 63.3), with higher SAPS-I and SOFA, more MICU (49.6% of deaths) and fewer CSRU (7.8%). Height is missing for 1909/4000. Length of stay is an outcome, not a predictor.

`output/outcomes.csv` summarizes the same endpoints. Crude mortality is 13.85%. Mean SAPS-I is 14.2 and mean SOFA is 6.4; both are higher among deaths. These scores stay in the outcome file for characterization only.

## 3. Missingness

`output/missingness_map.png` has two panels: `missingno.matrix` on admission descriptors (white = missing) and a bar chart of stays that never have each time-series variable. Height is missing for about half of stays. Vitals and routine labs are dense. Troponin, cholesterol, and `RespRate` are sparse. Troponin I and T never co-occur. `MechVent` is recorded only as 1 and never with `RespRate`.

`output/missingness_vs_death.csv` compares mortality when a variable was measured vs never measured. Troponin I: 27.8% vs 13.1% dead. Liver enzymes and lactate are also higher when ordered. `RespRate` goes the other way: 7.9% if measured vs 16.1% if not (unmeasured ≈ ventilated). Missingness is informative. Do not impute a normal respiratory rate; keep missingness indicators.

## 4. AI-use statement

Cursor Agent (Grok 4.6) helped write `checkpoint1.py`, merge the 4000 files, and draft plots. I set the directory layout, physiologic bounds, and the 24h split; I checked the 4000-file count, record 132539, Height `-1` counts, and the Troponin / RespRate overlap. Faster merge; bounds are mine, not from the spec. No death model was trained.
