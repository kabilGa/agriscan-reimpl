| backbone | split | variant | seeds (n) | val acc (mean, min-max) | macro recall | role |
|---|---|---|---|---|---|---|
| efficientnet_b3 | clean | base | 3 | 0.9976 (0.9974-0.9978) | 0.9975 | core, seed_replicate |
| efficientnet_b3 | original | base | 1 | 0.9970 (single run) | 0.9968 | core |
| mobilenet_v2 | clean | base | 3 | 0.9909 (0.9898-0.9925) | 0.9908 | core, seed_replicate |
| mobilenet_v2 | clean | frozen_only | 1 | 0.9373 (single run) | 0.9372 | ablation |
| mobilenet_v2 | clean | no_aug | 1 | 0.9965 (single run) | 0.9965 | ablation |
| mobilenet_v2 | clean | sched_constant | 1 | 0.9813 (single run) | 0.9811 | ablation |
| mobilenet_v2 | clean | sched_step | 1 | 0.9924 (single run) | 0.9923 | ablation |
| mobilenet_v2 | clean | size128 | 1 | 0.9658 (single run) | 0.9657 | ablation |
| mobilenet_v2 | clean | size160 | 1 | 0.9856 (single run) | 0.9856 | ablation |
| mobilenet_v2 | original | base | 1 | 0.9917 (single run) | 0.9915 | core |
| resnet50 | clean | base | 1 | 0.9961 (single run) | 0.9961 | extra |

| run | acc | ECE | train min | fwd img/s (device) | size MB | date | commit |
|---|---|---|---|---|---|---|---|
| efficientnet_b3__clean__s0__base | 0.9977 | 0.0010 | 31.7 | 371 (cuda) | 43.0 | 2026-10-15 | ace8631 |
| efficientnet_b3__clean__s1__base | 0.9978 | 0.0008 | 31.7 | 371 (cuda) | 43.0 | 2026-10-15 | ace8631 |
| efficientnet_b3__clean__s2__base | 0.9974 | 0.0010 | 31.8 | 371 (cuda) | 43.0 | 2026-10-15 | ace8631 |
| efficientnet_b3__original__s0__base | 0.9970 | 0.0005 | 31.7 | 369 (cuda) | 43.0 | 2026-10-15 | ace8631 |
| mobilenet_v2__clean__s0__base | 0.9904 | 0.0023 | 18.0 | 2100 (cuda) | 9.1 | 2026-10-15 | ace8631 |
| mobilenet_v2__clean__s0__frozen_only | 0.9373 | 0.0055 | 18.2 | 2109 (cuda) | 9.1 | 2026-10-15 | ace8631 |
| mobilenet_v2__clean__s0__no_aug | 0.9965 | 0.0013 | 14.1 | 2070 (cuda) | 9.1 | 2026-10-15 | ace8631 |
| mobilenet_v2__clean__s0__sched_constant | 0.9813 | 0.0038 | 18.8 | 2100 (cuda) | 9.1 | 2026-10-15 | ace8631 |
| mobilenet_v2__clean__s0__sched_step | 0.9924 | 0.0018 | 18.5 | 2095 (cuda) | 9.1 | 2026-10-15 | ace8631 |
| mobilenet_v2__clean__s0__size128 | 0.9658 | 0.0026 | 11.1 | 4378 (cuda) | 9.1 | 2026-10-15 | ace8631 |
| mobilenet_v2__clean__s0__size160 | 0.9856 | 0.0040 | 13.4 | 3670 (cuda) | 9.1 | 2026-10-15 | ace8631 |
| mobilenet_v2__clean__s1__base | 0.9925 | 0.0020 | 18.2 | 2095 (cuda) | 9.1 | 2026-10-15 | ace8631 |
| mobilenet_v2__clean__s2__base | 0.9898 | 0.0017 | 18.3 | 2088 (cuda) | 9.1 | 2026-10-15 | ace8631 |
| mobilenet_v2__original__s0__base | 0.9917 | 0.0015 | 18.7 | 2097 (cuda) | 9.1 | 2026-10-15 | ace8631 |
| resnet50__clean__s0__base | 0.9961 | 0.0011 | 25.4 | 801 (cuda) | 94.3 | 2026-10-15 | ace8631 |
