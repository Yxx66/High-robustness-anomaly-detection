# Ablation Core (a–e) — SGDClassifier

| group | run_id | train_regime_used | adversarial_method | feature_selection | f1__clean | f1__fgsm | f1__pgd | f1__transfer_pgd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a | a_no_fs_no_aug | baseline | none |  | 0.9775 | 0.1276 | 0.1287 | 0.1489 |
| b | b_fs_forward_only | baseline | none | forward | 0.9670 | 0.9670 | 0.9670 | 0.9670 |
| c | c_fgsm_only | adv_train | fgsm |  | 0.9745 | 0.9988 | 0.9963 | 0.6600 |
| d | d_owc_only | adv_train | owc-sawn |  | 0.9782 | 0.1791 | 0.1677 | 0.2468 |
| e | e_fgsm_owc | adv_train | both |  | 0.9748 | 0.9975 | 0.9937 | 0.6026 |
