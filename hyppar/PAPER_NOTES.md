# HypPAR paper notes (UPAR Track 2)

## Hypothesis

Do parameter-matched Lorentz **entailment** embeddings improve compositional pedestrian attribute retrieval vs Euclidean twins under the official UPAR Track 2 protocol?

## Contribution framing

1. **Hyperbolic entailment retrieval** for attribute queries: sparse queries near the origin (general), richer images deeper (specific); gallery should entail the query (MERU-style cones).
2. Category-aware Lorentz attribute prototypes + radius-aware composition over the 40-bit UPAR vocabulary.
3. Strong PAR head for attribute BCE, with geometry isolated via Euclidean twins / score-mode ablations (`entailment` | `distance` | `attr_l1` | `hybrid`).

## Clean protocol (mandatory for reported tables)

From-scratch curriculum via `make train-main` / `hyppar.scripts.train_curriculum`:

- Stage A: frozen ImageNet backbone  
- Stage B: unfreeze (full or last stages), continue from Stage A only  
- No warm-start from exploratory checkpoints  

## Related work

- UPAR / UPAR challenges (Specker, Cormier et al.)
- MERU ([2304.09172](https://arxiv.org/abs/2304.09172))
- Compositional entailment hyperbolic VLMs ([2410.06912](https://arxiv.org/abs/2410.06912))
- HiHR hyperbolic ReID — differentiate: identity ReID vs attribute-query retrieval

## Official metric

Primary: **mADM**. Secondary: mAP, Rank-1; per-domain; by query cardinality.

## Exploratory val notes (pre-clean curriculum)

| Setting | mADM | mAP | R-1 |
| --- | ---: | ---: | ---: |
| Frozen prototypes | ~33 | ~4 | ~5 |
| Partial FT Lorentz | ~43 | ~8 | ~9 |
| Entailment FT | ~45 | ~8.5 | ~10.5 |
| Hybrid score (same ckpt) | ~48 | ~8.4 | ~10.6 |
| Codabench test (pre-entail FT) | 34.9 | 12.6 | 12.5 |

Re-run with clean curriculum before camera-ready tables.

## Codabench

`make package` → `checkpoints/hyppar_task2_submission.zip` (`run.py` at zip root).
