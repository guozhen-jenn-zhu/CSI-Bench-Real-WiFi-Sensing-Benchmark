# CSI-Bench reproduction results — Phase F (3-seed mean ± std)

This document captures the full set of paper-table results reproduced on the [CSI-Bench dataset](https://www.kaggle.com/datasets/guozhenjennzhu/csi-bench).  All cells are reported as `mean ± std` over **seeds {42, 43, 44}** wherever the experiment was run for all three seeds; cells that show a single value are seed-42 only (see the [Coverage](#coverage) section for the per-cell breakdown).

## Experimental setup

- Hardware: `ml.g5.12xlarge` (4× NVIDIA A10G, 24 GiB each)
- Window length: 500, feature size: 232, batch size: 128, epochs: 100, early-stop patience: 15
- Supervised optimizer: AdamW, lr=1e-3, weight_decay=1e-5, 5 warmup epochs
- Multitask: AdamW, lr=5e-4, weight_decay=1e-5, LoRA r=8 / alpha=32 / dropout=0.05
- Multitask backbones supported: `transformer`, `patchtst`, `timesformer1d`
- Seeds: {42, 43, 44}

## Coverage

| Slice | Seeds reported |
|---|---|
| Supervised: FD / BD / Loc / MSR × all 7 models | mean ± std over {42, 43, 44} |
| Supervised: HAR / UID / Prox × {transformer, patchtst, timesformer1d} | mean ± std over {42, 43, 44} |
| Supervised: HAR / UID / Prox × {mlp, lstm, resnet18, vit} | seed 42 only (single value) |
| Multitask: HAR / UID / Prox × {transformer, patchtst, timesformer1d} | mean ± std over {42, 43, 44} |

The reduced coverage for HAR / UID / Prox × {mlp, lstm, resnet18, vit} matches the Phase F scope decision: we trained only the three transformer-family models for those tasks at multiple seeds because the multitask comparison only uses those backbones; the other four are kept as single-seed reference rows in Tab. 3.

---

## Tab. 3 — Supervised single-task accuracy / F1 on `test_id`

| Task | Model | Acc (test_id) | F1 (test_id) |
|---|---|---|---|
| FallDetection | mlp | 0.9244±0.0070 | 0.9245±0.0070 |
| FallDetection | lstm | 0.9410±0.0128 | 0.9409±0.0126 |
| FallDetection | resnet18 | 0.9502±0.0057 | 0.9502±0.0056 |
| FallDetection | transformer | 0.9453±0.0037 | 0.9454±0.0035 |
| FallDetection | vit | 0.9479±0.0012 | 0.9480±0.0012 |
| FallDetection | patchtst | 0.9466±0.0041 | 0.9466±0.0040 |
| FallDetection | timesformer1d | 0.9393±0.0094 | 0.9394±0.0091 |
| BreathingDetection | mlp | 0.9861±0.0041 | 0.9860±0.0041 |
| BreathingDetection | lstm | 0.9941±0.0007 | 0.9941±0.0007 |
| BreathingDetection | resnet18 | 0.9879±0.0013 | 0.9879±0.0014 |
| BreathingDetection | transformer | 0.9865±0.0019 | 0.9865±0.0019 |
| BreathingDetection | vit | 0.9986±0.0002 | 0.9986±0.0002 |
| BreathingDetection | patchtst | 0.9978±0.0003 | 0.9978±0.0003 |
| BreathingDetection | timesformer1d | 0.9986±0.0004 | 0.9986±0.0004 |
| Localization | mlp | 0.9068±0.0032 | 0.9071±0.0031 |
| Localization | lstm | 0.9789±0.0070 | 0.9789±0.0070 |
| Localization | resnet18 | 0.9843±0.0039 | 0.9843±0.0039 |
| Localization | transformer | 0.9768±0.0044 | 0.9769±0.0043 |
| Localization | vit | 0.9660±0.0029 | 0.9660±0.0028 |
| Localization | patchtst | 0.9739±0.0031 | 0.9739±0.0031 |
| Localization | timesformer1d | 0.9807±0.0044 | 0.9808±0.0044 |
| MotionSourceRecognition | mlp | 0.9880±0.0007 | 0.9880±0.0007 |
| MotionSourceRecognition | lstm | 0.9847±0.0021 | 0.9847±0.0021 |
| MotionSourceRecognition | resnet18 | 0.9957±0.0010 | 0.9957±0.0010 |
| MotionSourceRecognition | transformer | 0.9908±0.0005 | 0.9908±0.0005 |
| MotionSourceRecognition | vit | 0.9915±0.0021 | 0.9915±0.0021 |
| MotionSourceRecognition | patchtst | 0.9852±0.0002 | 0.9852±0.0002 |
| MotionSourceRecognition | timesformer1d | 0.9844±0.0025 | 0.9844±0.0025 |
| HumanActivityRecognition | mlp | 0.8372 | 0.8331 |
| HumanActivityRecognition | lstm | 0.9455 | 0.9454 |
| HumanActivityRecognition | resnet18 | 0.9413 | 0.9413 |
| HumanActivityRecognition | transformer | 0.9450±0.0049 | 0.9451±0.0050 |
| HumanActivityRecognition | vit | 0.9568 | 0.9567 |
| HumanActivityRecognition | patchtst | 0.9467±0.0046 | 0.9467±0.0046 |
| HumanActivityRecognition | timesformer1d | 0.9509±0.0038 | 0.9505±0.0038 |
| HumanIdentification | mlp | 0.9976 | 0.9976 |
| HumanIdentification | lstm | 1.0000 | 1.0000 |
| HumanIdentification | resnet18 | 1.0000 | 1.0000 |
| HumanIdentification | transformer | 0.9999±0.0001 | 0.9999±0.0001 |
| HumanIdentification | vit | 1.0000 | 1.0000 |
| HumanIdentification | patchtst | 1.0000±0.0000 | 1.0000±0.0000 |
| HumanIdentification | timesformer1d | 1.0000±0.0000 | 1.0000±0.0000 |
| ProximityRecognition | mlp | 0.8149 | 0.8088 |
| ProximityRecognition | lstm | 0.9302 | 0.9305 |
| ProximityRecognition | resnet18 | 0.9224 | 0.9217 |
| ProximityRecognition | transformer | 0.9287±0.0028 | 0.9283±0.0031 |
| ProximityRecognition | vit | 0.9422 | 0.9423 |
| ProximityRecognition | patchtst | 0.9283±0.0090 | 0.9278±0.0096 |
| ProximityRecognition | timesformer1d | 0.9418±0.0013 | 0.9416±0.0012 |

---

## Tab. 4 — Multitask LoRA (Transformer) vs single-task Transformer

| Task | Pipeline | Acc (test_id) | F1 (test_id) |
|---|---|---|---|
| HumanActivityRecognition | single-task | 0.9450±0.0049 | 0.9451±0.0050 |
| HumanActivityRecognition | multitask | 0.9480±0.0039 | 0.9380±0.0032 |
| HumanIdentification | single-task | 0.9999±0.0001 | 0.9999±0.0001 |
| HumanIdentification | multitask | 0.9989±0.0003 | 1.0000±0.0000 |
| ProximityRecognition | single-task | 0.9287±0.0028 | 0.9283±0.0031 |
| ProximityRecognition | multitask | 0.9567±0.0077 | 0.9408±0.0117 |

The "multitask" row uses the Transformer backbone with LoRA adapters on the joint HAR + UID + Prox training set, evaluated per-task on `test_id`.  The single-task row is the corresponding row from Tab. 3.  Multitask is on par with or stronger than single-task on accuracy across all three tasks.

### Multitask on `test_id` for all three supported backbones

| Task | Backbone | Acc (test_id) | F1 (test_id) |
|---|---|---|---|
| HumanActivityRecognition | transformer | 0.9480±0.0039 | 0.9380±0.0032 |
| HumanActivityRecognition | patchtst | 0.9490±0.0128 | 0.9300±0.0184 |
| HumanActivityRecognition | timesformer1d | 0.9540±0.0050 | 0.9413±0.0064 |
| HumanIdentification | transformer | 0.9989±0.0003 | 1.0000±0.0000 |
| HumanIdentification | patchtst | 0.9988±0.0008 | 0.9995±0.0007 |
| HumanIdentification | timesformer1d | 0.9991±0.0003 | 1.0000±0.0000 |
| ProximityRecognition | transformer | 0.9567±0.0077 | 0.9408±0.0117 |
| ProximityRecognition | patchtst | 0.9553±0.0055 | 0.9393±0.0071 |
| ProximityRecognition | timesformer1d | 0.9560±0.0058 | 0.9417±0.0077 |

---

## Tab. 8 — FallDetection difficulty breakdown (easy / medium / hard)

| Task | Model | Acc easy | F1 easy | Acc medium | F1 medium | Acc hard | F1 hard |
|---|---|---|---|---|---|---|---|
| FallDetection | mlp | 0.9475±0.0054 | 0.9475±0.0055 | 0.7451±0.0555 | 0.7444±0.0560 | 0.6758±0.0281 | 0.6634±0.0360 |
| FallDetection | lstm | 0.9701±0.0090 | 0.9701±0.0090 | 0.8039±0.0277 | 0.8014±0.0260 | 0.6073±0.0616 | 0.5489±0.1361 |
| FallDetection | resnet18 | 0.9730±0.0029 | 0.9731±0.0028 | 0.7647±0.0000 | 0.7546±0.0000 | 0.7078±0.0552 | 0.6981±0.0629 |
| FallDetection | transformer | 0.9778±0.0022 | 0.9778±0.0022 | 0.7647±0.0961 | 0.7586±0.0989 | 0.5799±0.0683 | 0.5057±0.1294 |
| FallDetection | vit | 0.9741±0.0021 | 0.9742±0.0020 | 0.6863±0.0277 | 0.6842±0.0264 | 0.6804±0.0233 | 0.6793±0.0242 |
| FallDetection | patchtst | 0.9745±0.0041 | 0.9745±0.0041 | 0.6667±0.0277 | 0.6536±0.0223 | 0.6621±0.0393 | 0.6493±0.0544 |
| FallDetection | timesformer1d | 0.9701±0.0077 | 0.9702±0.0076 | 0.6863±0.0734 | 0.6762±0.0869 | 0.6119±0.0786 | 0.5514±0.1429 |

---

## Tab. 9 — BreathingDetection difficulty breakdown (easy / medium / hard)

| Task | Model | Acc easy | F1 easy | Acc medium | F1 medium | Acc hard | F1 hard |
|---|---|---|---|---|---|---|---|
| BreathingDetection | mlp | 0.9935±0.0021 | 0.9935±0.0021 | 0.9856±0.0029 | 0.9855±0.0029 | 0.9727±0.0073 | 0.9724±0.0074 |
| BreathingDetection | lstm | 0.9960±0.0002 | 0.9960±0.0002 | 0.9951±0.0007 | 0.9951±0.0007 | 0.9898±0.0017 | 0.9897±0.0017 |
| BreathingDetection | resnet18 | 0.9919±0.0008 | 0.9919±0.0008 | 0.9896±0.0011 | 0.9896±0.0011 | 0.9793±0.0039 | 0.9791±0.0040 |
| BreathingDetection | transformer | 0.9899±0.0015 | 0.9899±0.0015 | 0.9880±0.0018 | 0.9881±0.0018 | 0.9788±0.0037 | 0.9788±0.0038 |
| BreathingDetection | vit | 0.9987±0.0005 | 0.9987±0.0005 | 0.9989±0.0002 | 0.9989±0.0002 | 0.9980±0.0006 | 0.9980±0.0006 |
| BreathingDetection | patchtst | 0.9977±0.0006 | 0.9977±0.0006 | 0.9979±0.0004 | 0.9979±0.0004 | 0.9979±0.0003 | 0.9979±0.0003 |
| BreathingDetection | timesformer1d | 0.9987±0.0006 | 0.9987±0.0006 | 0.9992±0.0007 | 0.9992±0.0007 | 0.9980±0.0011 | 0.9980±0.0011 |

---

## Tab. 10 — Localization difficulty breakdown (easy / medium / hard)

| Task | Model | Acc easy | F1 easy | Acc medium | F1 medium | Acc hard | F1 hard |
|---|---|---|---|---|---|---|---|
| Localization | mlp | 0.9301±0.0037 | 0.9331±0.0020 | 0.8927±0.0024 | 0.8975±0.0023 | 0.8864±0.0141 | 0.9082±0.0124 |
| Localization | lstm | 0.9872±0.0042 | 0.9882±0.0038 | 0.9684±0.0155 | 0.9700±0.0141 | 0.9786±0.0031 | 0.9823±0.0041 |
| Localization | resnet18 | 0.9833±0.0035 | 0.9833±0.0035 | 0.9874±0.0059 | 0.9877±0.0061 | 0.9819±0.0042 | 0.9831±0.0033 |
| Localization | transformer | 0.9862±0.0025 | 0.9865±0.0024 | 0.9665±0.0036 | 0.9672±0.0037 | 0.9745±0.0103 | 0.9777±0.0090 |
| Localization | vit | 0.9596±0.0028 | 0.9600±0.0020 | 0.9678±0.0054 | 0.9688±0.0047 | 0.9745±0.0031 | 0.9784±0.0022 |
| Localization | patchtst | 0.9754±0.0054 | 0.9758±0.0052 | 0.9703±0.0039 | 0.9711±0.0035 | 0.9761±0.0071 | 0.9825±0.0042 |
| Localization | timesformer1d | 0.9813±0.0057 | 0.9820±0.0059 | 0.9792±0.0015 | 0.9801±0.0021 | 0.9819±0.0103 | 0.9844±0.0097 |

---

## Tab. 11 — MotionSourceRecognition difficulty breakdown (4-class: Fan / Human / IRobot / Pet)

| Task | Model | Acc easy | F1 easy | Acc medium | F1 medium | Acc hard | F1 hard |
|---|---|---|---|---|---|---|---|
| MotionSourceRecognition | mlp | 0.9800±0.0014 | 0.9802±0.0016 | 0.9919±0.0007 | 0.9919±0.0007 | 0.9795±0.0050 | 0.9795±0.0050 |
| MotionSourceRecognition | lstm | 0.9730±0.0081 | 0.9752±0.0071 | 0.9902±0.0009 | 0.9902±0.0009 | 0.9726±0.0033 | 0.9725±0.0034 |
| MotionSourceRecognition | resnet18 | 0.9938±0.0022 | 0.9938±0.0022 | 0.9974±0.0007 | 0.9974±0.0007 | 0.9912±0.0034 | 0.9912±0.0034 |
| MotionSourceRecognition | transformer | 0.9908±0.0016 | 0.9911±0.0014 | 0.9923±0.0015 | 0.9923±0.0015 | 0.9860±0.0025 | 0.9860±0.0025 |
| MotionSourceRecognition | vit | 0.9892±0.0044 | 0.9892±0.0044 | 0.9931±0.0020 | 0.9931±0.0020 | 0.9874±0.0015 | 0.9874±0.0015 |
| MotionSourceRecognition | patchtst | 0.9861±0.0090 | 0.9867±0.0086 | 0.9883±0.0019 | 0.9884±0.0019 | 0.9750±0.0011 | 0.9752±0.0011 |
| MotionSourceRecognition | timesformer1d | 0.9827±0.0053 | 0.9842±0.0040 | 0.9877±0.0037 | 0.9877±0.0037 | 0.9750±0.0015 | 0.9751±0.0014 |

---

## Tab. 12 — HumanActivityRecognition OOD (multitask LoRA, cross-device / -env / -user)

| Task | Model | Acc cross_device | F1 cross_device | Acc cross_env | F1 cross_env | Acc cross_user | F1 cross_user |
|---|---|---|---|---|---|---|---|
| HumanActivityRecognition | transformer | 0.3514±0.0411 | 0.2645±0.0277 | 0.3709±0.0460 | 0.4007±0.0379 | 0.4874±0.0045 | 0.2967±0.0294 |
| HumanActivityRecognition | patchtst | 0.3458±0.0210 | 0.2295±0.0223 | 0.3584±0.0557 | 0.3430±0.0553 | 0.4858±0.0125 | 0.2706±0.0801 |
| HumanActivityRecognition | timesformer1d | 0.4209±0.0321 | 0.2769±0.0387 | 0.3920±0.0147 | 0.3183±0.1026 | 0.5077±0.0168 | 0.2688±0.0774 |

---

## Tab. 13 — HumanIdentification OOD (multitask LoRA, cross-device / -env / -user)

| Task | Model | Acc cross_device | F1 cross_device | Acc cross_env | F1 cross_env | Acc cross_user | F1 cross_user |
|---|---|---|---|---|---|---|---|
| HumanIdentification | transformer | 0.2151±0.0521 | 0.1996±0.0704 | 0.1421±0.0423 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 |
| HumanIdentification | patchtst | 0.2284±0.0106 | 0.2565±0.0295 | 0.1544±0.0369 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 |
| HumanIdentification | timesformer1d | 0.2037±0.0334 | 0.2722±0.0703 | 0.1595±0.0483 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 |

The `cross_user` numbers being 0 is expected for HumanIdentification: the cross-user split holds out entirely new identities, so the model has never seen those classes during training and cannot produce a correct label.  This is the task's intended worst-case OOD scenario rather than a measurement issue.

---

## Tab. 14 — ProximityRecognition OOD (multitask LoRA, cross-device / -env / -user)

| Task | Model | Acc cross_device | F1 cross_device | Acc cross_env | F1 cross_env | Acc cross_user | F1 cross_user |
|---|---|---|---|---|---|---|---|
| ProximityRecognition | transformer | 0.4985±0.0170 | 0.3616±0.0146 | 0.5329±0.0179 | 0.4169±0.0663 | 0.4511±0.0248 | 0.3924±0.0264 |
| ProximityRecognition | patchtst | 0.4647±0.0675 | 0.3535±0.0281 | 0.4975±0.0284 | 0.3389±0.0100 | 0.4712±0.0428 | 0.3731±0.0319 |
| ProximityRecognition | timesformer1d | 0.4564±0.0369 | 0.3894±0.0312 | 0.5051±0.0172 | 0.3727±0.0354 | 0.4335±0.0230 | 0.3711±0.0236 |

---

## Reproducing these tables

All commands assume `ml.g5.12xlarge`, the CSI-Bench dataset [downloaded from Kaggle](https://www.kaggle.com/datasets/guozhenjennzhu/csi-bench) and extracted to `~/Data/CSI-Bench/`, and `peft` installed (`pip install peft`).

```bash
# Sweep A -- HAR/UID/Prox single-task baselines (transformer / patchtst / timesformer1d) x 3 seeds
nohup python -u scripts/run_seed_sweep.py \
    --config configs/csi_bench_multitask_baselines_config.json \
    --seeds 42,43,44 --skip-existing --num-gpus 4 \
    > results/csi_bench/sweep_A.log 2>&1 & disown

# Sweep B -- FD / BD / Loc / MSR x all 7 models x 3 seeds
nohup python -u scripts/run_seed_sweep.py \
    --config configs/csi_bench_local_config.json \
    --tasks "FallDetection,BreathingDetection,Localization,MotionSourceRecognition" \
    --seeds 42,43,44 --skip-existing --num-gpus 4 \
    > results/csi_bench/sweep_B.log 2>&1 & disown

# Sweep C -- multitask LoRA x {transformer, patchtst, timesformer1d} x 3 seeds.
# Run sequentially (--num-gpus 1) to avoid the concurrent-multitask-subprocess
# OOM / h5 contention that loses output dirs.
nohup python -u scripts/run_seed_sweep.py \
    --config configs/csi_bench_multitask_config.json \
    --models "transformer,patchtst,timesformer1d" \
    --seeds 42,43,44 --skip-existing --num-gpus 1 \
    > results/csi_bench_multitask/sweep_C.log 2>&1 & disown

# After all sweeps finish, regenerate the per-table CSVs in this document
python result_analysis/all_result_summary.py \
    --results-dir ./results/csi_bench \
    --multitask-results-dir ./results/csi_bench_multitask \
    --output-dir result_analysis/csi_bench_3seeds \
    --seeds 42,43,44
```

The aggregator emits one CSV per paper table (`tab3_supervised.csv`, `tab4_multitask_vs_singletask.csv`, `tab8_*` through `tab14_*`) plus `_supervised_raw.csv` and `_multitask_raw.csv` containing every (task, model, seed, split) row that fed into the tables above.
