# Student model — distilled from the teacher's out-of-fold predictions

`XGBRegressor(objective="reg:logistic", n_estimators=100, max_depth=4)` — 100 trees vs. the teacher's 5x500, trained to regress against `teacher-xgboost`'s leak-free out-of-fold attack probability (not the true label directly — this is distillation, not a second from-scratch classifier). Held out via the same `StratifiedGroupKFold` grouping as the teacher, so this evaluation is on feature-patterns the student's own training never saw.

## Distillation fidelity (student vs. teacher, same held-out rows)

- RMSE (student probability vs. teacher OOF probability): 0.0230
- Verdict-bucket agreement (allow/uncertain/block, student vs. teacher): 0.9956

## Real-world correctness (both models vs. true label, same held-out rows)

| | Teacher (OOF) | Student (distilled) |
|---|---|---|
| Accuracy | 0.9969 | 0.9966 |
| Precision | 0.9825 | 0.9837 |
| Recall | 0.9899 | 0.9861 |
| F1 | 0.9862 | 0.9849 |

**Train/production-drift caveat, per `knowledge-graph/noxos-inference/TASKS.md`**: the teacher numbers above come from 5 separate fold-models' OOF predictions, not a single full-data production teacher — don't assume they're interchangeable artifacts once a real production teacher is trained and deployed.
