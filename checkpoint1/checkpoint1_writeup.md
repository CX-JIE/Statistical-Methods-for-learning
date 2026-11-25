# Checkpoint 1 write-up

Junjie Cai (jc1206), BIOSTAT 707

`pixi run --locked checkpoint1` (from `checkpoint1/` or from the repository root) reads `data/set-a/`
(4,000 files, one per intensive care unit stay) and `data/Outcomes-a.txt`, and writes every file
referred to below to `output/`.

Abbreviations: ICU, intensive care unit; CCU, coronary care unit; CSRU, cardiac surgery recovery
unit; MICU, medical ICU; SICU, surgical ICU; GCS, Glasgow Coma Scale; SAPS-I, Simplified Acute
Physiology Score I; SOFA, Sequential Organ Failure Assessment; FiO2, fraction of inspired oxygen;
BUN, blood urea nitrogen; SD, standard deviation.

## 1. What information exists at prediction time

The Challenge 2012 prediction is made at the end of the first 48 hours of an ICU stay. At that
point we know:

* **Admission descriptors**, recorded at time 00:00: `Age`, `Gender`, `Height`, `ICUType` and
  admission `Weight`.
* **Time-stamped measurements** of 36 time-series variables (vital signs, laboratory values,
  urine output, `MechVent`) taken during hours 0 to 48.
* **Whether and how often each variable was measured.** This is known at hour 48 and, as section 3
  shows, carries information of its own.

We do **not** know, and must never use as predictors:

* `In-hospital_death`, the outcome.
* `Length_of_stay` and `Survival`. Both describe what happened after the 48-hour window, so using
  them would be leakage.
* `SAPS-I` and `SOFA` from `Outcomes-a.txt`. They are published severity scores kept for
  description here and as benchmarks later, not as model inputs.

**Files.** `output/set-a_long.csv` is the long table: 1,757,980 rows, one per measurement, with
`RecordID`, `Time`, `Parameter` and `Value`. It keeps the raw values exactly as recorded
(including the `-1` codes) so that every later step can be audited against it.
`output/set-a_wide.csv` is the wide table: 4,000 rows, one per stay, 435 columns. For each
time-series variable it gives the count, first, last, minimum, maximum and mean of the cleaned
values over hours 0 to 48, plus the total urine volume (`Urine_sum`). For the 17 variables measured
most often (median of at least 4 values per stay, for example heart rate, temperature, GCS and blood
pressure) the same summaries are also given separately for hours 0 to 24 and 24 to 48, so that a
later model can see the direction of change. The outcomes are joined to the wide table for
description only.

**Cleaning decisions** (applied to the wide table and the missingness files, not to the long table):

* `-1` in an admission descriptor means "not recorded" and was set to missing (Height 1,894 stays,
  Weight 326, Gender 3). A height of 0 was also set to missing.
* Negative time-series values were set to missing.
* FiO2 of 0 was set to missing (below room air is impossible); FiO2 values between 1 and 100 were
  recorded in percent and were divided by 100.
* Height recorded in metres (1.2 to 2.5) or inches (40 to 90) was converted to centimetres, and
  only 100 to 250 cm was kept. Weight was kept between 20 and 300 kg and age between 15 and 120
  years.
* Physiologically implausible values were removed using variable-specific bounds (for example heart
  rate 20 to 250 beats per minute, temperature 30 to 45 degrees C, pH 6.5 to 8.0). The bounds are
  my own clinical choices, not given in the assignment, and are listed in `PHYSIO_BOUNDS` in
  `checkpoint1.py`.
* Admission weight is the value at 00:00 only; later weights mostly reflect fluid given in the ICU.

## 2. Table 1 and outcome summary

`output/table1.csv` describes the 4,000 stays overall and by outcome: **554 died in hospital
(13.85%) and 3,446 survived.** Means are given with their SD; p-values are the TableOne defaults
(comparison of means for continuous variables, chi-square test for categorical ones) and are
descriptive, with no adjustment for multiple comparisons.

* **Age.** Patients who died were older: mean 70.0 (SD 16.1) against 63.3 (17.6) years, p < 0.001.
* **ICU type** differs clearly (p < 0.001). The MICU accounts for 49.6% of deaths but 35.0% of
  survivors, while the CSRU accounts for only 7.8% of deaths but 24.1% of survivors. The CCU and
  SICU shares are similar in both groups (14.6% vs 14.4% and 28.0% vs 26.5%).
* **Sex** does not differ (54.2% male among deaths, 56.5% among survivors, p = 0.379). Sex is not
  recorded for 3 stays.
* **Weight** is slightly lower among deaths (78.5 vs 82.0 kg, p = 0.002; missing for 326 stays).
  **Height** does not differ (168.7 vs 169.8 cm, p = 0.129) and is missing for 1,909 stays (47.7%),
  so it is a weak descriptor.
* **Length of stay** is 14.5 against 13.3 days (p = 0.073). It is an outcome, shown only to
  describe the cohort.

In short, the patients who died were older, more often medical ICU patients and less often
cardiac-surgery patients, who are a low-risk, planned-admission group. The small weight difference
is most likely a case-mix effect and should not be read causally.

`output/outcomes.csv` summarises the endpoints: crude in-hospital mortality is 13.85% and the mean
length of stay is 13.4 days. Both severity scores are higher among deaths (SAPS-I 16.6 vs 13.8;
SOFA 8.3 vs 6.1 as reported in the file and in Table 1), which confirms that they carry signal and
makes them sensible benchmarks.

**Caveat on these two scores.** In `Outcomes-a.txt`, SAPS-I is coded `-1` (not computed) for 190
stays and SOFA for 144 stays. Table 1 and `outcomes.csv` include these `-1` values as they are, so
the reported means are slightly too low and Table 1's "Missing" column shows 0 for both scores.
With the `-1` codes excluded, mean SAPS-I is 14.96 (17.40 in deaths, 14.57 in survivors) and mean
SOFA is 6.68 (8.43 vs 6.39). The direction and size of the difference between deaths and
survivors are unchanged. From Checkpoint 2 on, these codes are treated as missing.

## 3. Missingness

`output/missingness_map.png` has two panels. The top panel is a `missingno` matrix of the
admission descriptors (white = missing): height is missing for about half of the stays, weight for
8%, and age, sex and ICU type are essentially complete. The bottom panel is a bar chart of the
percentage of stays in which each time-series variable was **never** measured in hours 0 to 48;
red bars mark variables never measured in at least 50% of stays.

The variables fall into three groups:

* **Dense** (never measured in under 3% of stays): heart rate, temperature, GCS, BUN, creatinine,
  haematocrit, sodium, potassium, magnesium, bicarbonate, glucose, platelets, white cell count and
  urine output. Only 3 stays have no time-series value at all.
* **Intermediate**: non-invasive blood pressure (never measured in about 13%), blood gases (pH,
  PaO2 and PaCO2, about 24%), invasive blood pressure (about 30%), FiO2 (32%), `MechVent` (37%)
  and lactate (45%).
* **Sparse** (never measured in at least 50%): troponin I (94.9%), cholesterol (92.4%), troponin T
  (78.4%), respiratory rate (72.5%), albumin (59.6%), the liver panel (ALP, ALT, AST and bilirubin,
  about 57% each) and SaO2 (55.2%).

Two structural patterns matter. **Troponin I and troponin T never occur in the same stay** (0
stays have both), which points to different assays used in different units rather than to
anything about the patient. **`MechVent` is only ever recorded as 1**, so its absence means either
"not ventilated" or "not charted", and **respiratory rate and `MechVent` never occur in the same
stay** (0 stays have both; 370 stays have neither). Respiratory rate is charted for spontaneously
breathing patients, so its absence largely marks ventilated patients.

`output/missingness_vs_death.csv` compares in-hospital mortality between stays in which a
variable was measured and stays in which it was never measured:

* **Ordering a test goes with higher risk.** Troponin I: 27.8% mortality when measured against
  13.1% when not; troponin T 21.1% vs 11.9%; the liver panel about 19.1% vs 9.9%; albumin 18.8% vs
  10.5%; lactate 17.6% vs 9.3%; `MechVent` recorded 15.6% vs 10.8%.
* **Respiratory rate goes the other way:** 7.9% mortality when measured against 16.1% when not,
  consistent with an unmeasured respiratory rate marking ventilation.
* **Little difference** for urine output (13.8% vs 14.5%), SaO2 (14.8% vs 13.1%), cholesterol
  (15.1% vs 13.7%) and non-invasive blood pressure (about 14.0% vs 13.0%).
* For the dense laboratory values, the never-measured groups have lower mortality (about 5 to 8%),
  but they contain only 63 to 117 stays and are likely short or incompletely recorded stays, so
  these comparisons are unstable.

**Conclusion.** Missingness here is plausibly informative, not completely at random: whether a test
was ordered reflects the clinicians' suspicion of a problem (troponin for cardiac events, lactate
for shock, the liver panel for liver disease), and a missing respiratory rate marks mechanical
ventilation. For modelling, this means keeping "never measured" indicators and measurement counts
as predictors and not filling missing values with normal ones as if they were missing at random.
These are crude, unadjusted comparisons: they show association with monitoring decisions, not a
causal effect of measuring anything.

## 4. AI-use statement

**Tools.** Cursor Agent (Grok 4.6), used in September 2026 for the original script and write-up,
and Claude (Claude Opus 5.5), used on 24 November 2026 to review the repository and expand this
write-up.

**What they were used for.** Cursor Agent helped write `checkpoint1.py`, merge the 4,000 record
files and draft the plots. Claude drafted the expanded text of sections 1 to 3 from the files in
`output/` and the script, and added the fallback that lets the script find `data/` at the
repository root. I set the directory layout, the physiological bounds and the 0 to 24 / 24 to 48
hour split, and I decided what the write-up concludes.

**How the output was checked.** For the script, I checked the 4,000-file count, record 132539, the number of `-1` height codes, and the troponin and
respiratory-rate overlaps. For the expanded write-up, every number in sections 1 to 3 is taken
from the files in `output/` (or, for the `-1`-excluded score means, recomputed from
`Outcomes-a.txt`) and was checked against them, and the two co-occurrence statements in section 3
were recomputed from `output/set-a_long.csv`.

**Pros and cons.** The tools made the file merging, boilerplate and plotting much faster and gave a
useful second pass over the results. They do not know which clinical bounds are correct (the bounds
are mine, not from the assignment), and they can miss problems: the review in November found that
the `-1` codes in SAPS-I and SOFA had been left in the Table 1 and outcome means, which is now
stated in section 2. AI output therefore had to be checked against the data rather than accepted.
No mortality model was trained in this checkpoint.
