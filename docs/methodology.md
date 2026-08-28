# Methodology

## Research question

We test whether multi-subject pretraining provides a benefit for a held-out Subject 8 when the amount of Subject 8 fine-tuning data is reduced.

## Experimental design

Two initialization conditions are compared:

1. **Pretrained:** a checkpoint trained on Subjects 1–7, followed by Subject 8 fine-tuning.
2. **Scratch:** the same Subject 8 architecture trained from random initialization.

Both conditions use the same Subject 8 training subsets, hyperparameters, random seed policy, and fixed Subject 8 test set.

## Data budgets

The repository currently defines four explicit training budgets:

- 60 minutes: 750 samples
- 30 minutes: 375 samples
- 15 minutes: 188 samples
- 7.5 minutes: 94 samples

The mapping must be verified against the actual NSD/MindEye2 dataset before the first real run. The subsets should be nested so that the smallest set is contained in each larger set.

## Leakage prevention

Subject 8 must not appear in the pretraining checkpoint. The checkpoint path is intentionally left unset until it has been independently verified.

The test set is never truncated to create the training budgets and must remain identical across all eight runs.

## Evaluation

At minimum, save:

- forward retrieval top-1, top-5, and top-10;
- backward retrieval top-1, top-5, and top-10;
- reconstruction MSE;
- reconstruction Pearson correlation;
- reconstruction cosine similarity.

Each result must include the initialization condition and data budget.

## Interpretation

The central comparison is not simply the absolute best score. We will examine how quickly performance degrades as Subject 8 training data decreases, and whether pretraining gives a larger advantage at lower data budgets.
