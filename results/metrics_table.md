
# Metrics Summary


## baseline_clean_train_clean_test

| Model | Split | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|-------|----------|-----------|--------|----------|-----|
| SGDClassifier | clean_test | 95.94% | 96.47% | 99.07% | 97.75% | 95.80% |
| RidgeClassifier | clean_test | 95.41% | 95.91% | 99.07% | 97.46% | 94.71% |

## baseline_clean_train_adv_test

| Model | Split | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|-------|----------|-----------|--------|----------|-----|
| SGDClassifier | adv_test | 7.60% | 41.10% | 8.53% | 14.13% | 2.97% |
| RidgeClassifier | adv_test | 23.30% | 73.32% | 21.88% | 33.70% | 19.11% |

## adv_train_clean_test

| Model | Split | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|-------|----------|-----------|--------|----------|-----|
| SGDClassifier | clean_test | 95.48% | 96.07% | 98.98% | 97.50% | 94.99% |
| RidgeClassifier | clean_test | 94.66% | 95.32% | 98.87% | 97.06% | 92.35% |

## adv_train_adv_test

| Model | Split | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|-------|----------|-----------|--------|----------|-----|
| SGDClassifier | adv_test | 99.76% | 99.73% | 100.00% | 99.86% | 99.95% |
| RidgeClassifier | adv_test | 97.88% | 97.74% | 99.94% | 98.83% | 99.23% |