# Hill of Towie operator-response training data

These files contain the selected inputs and supervision for the audited run whose
update-1,500 checkpoint appears in hackathon commit
`f6e6450aa3adb90be1233a9258fbcd9edca1b710`. They contain 9,709 training,
172 validation, and 371 held-out examples. Each gzip-compressed JSONL file
preserves the order of the corresponding selected split, including the prompt,
answer options, exact target, turbine, cutoff, and model-input series. A null
sample marks a reading supplied as NaN to the model input adapter.

The run selected rows from the base continuation export followed by the audited
operator-response supplement. Selection used seed 42, caps of 240/3/8 per
category and label, and replay limits of 512/24/0 for train/validation/test.
The training count and category histogram were checked against the recorded run
log. Held-out turbine, cutoff, category, prompt, target, label, and row order
were checked against the archived evaluation inputs.

SCADA source: Clerc, A., and Lingkan, E. (2026), *Hill of Towie wind farm open
dataset*, version 2.1.0 (CC BY 4.0), released by RES on behalf of TRIG.
https://doi.org/10.5281/zenodo.22662930

The SCADA readings are sampled from the source and paired with hackathon-authored
questions and targets. This archive contains the dataset only.
