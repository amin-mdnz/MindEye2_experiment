# Experiment log

## Repository setup

- Repository: `amin-mdnz/MindEye2_experiment`
- Target subject: 8
- Conditions: pretrained Subjects 1–7 → Subject 8; scratch → Subject 8
- Budgets: 60m, 30m, 15m, 7.5m
- Test set: fixed across conditions

## Before first training run

- [ ] Verify MindEye2 dependencies and CUDA environment.
- [ ] Verify a forward pass.
- [ ] Verify the exact pretrained checkpoint.
- [ ] Confirm the checkpoint excludes Subject 8.
- [ ] Confirm the sample/minute mapping on the installed dataset.
- [ ] Confirm nested training indices.
- [ ] Confirm the fixed Subject 8 test indices.
- [ ] Run one small smoke test.

## Results

Results should be appended to `experiments/subject08_data_efficiency/results.csv` with condition and budget identifiers. Do not overwrite prior results without recording the reason.
