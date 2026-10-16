# PRISM

## Protocol-Aware Retrieval-Integrated Multimodal Sentiment Analysis with Missing Modalities

Research code accompanying the PRISM manuscript. PRISM studies sentiment prediction from textual, acoustic, and visual inputs when one or more modalities are unavailable.

The framework combines training-bank-driven missing-aware retrieval with protocol-specific fusion-head adaptation. It uses available evidence from the current sample together with same-modality training representations, and adapts the prediction head while keeping the learned feature backbone frozen.

## Release contents

The current repository distributes the source code as **`PRISM.zip`**. Download and extract this archive before running the commands below. Run all Python commands from the extracted directory containing `main.py`.

The archive also includes historical experiments and ablation scripts. Their presence does not mean that all optional modules are enabled in the final model. The formal runners listed below explicitly disable inactive experimental branches.

Some implementation names, including `IIE_model.py` and `IIEModel`, are inherited from the underlying codebase. The archive's existing `readme.md` contains historical instructions; use this repository-level README for the PRISM workflow.

**Not bundled in the current archive:** datasets, pretrained BERT weights, trained experiment checkpoints, and the original `paper_experiments/configs/` JSON files. Dataset preparation is required before training. Several historical smoke scripts expect those configuration files and cannot be used unchanged with this archive alone. The formal Python runners below define their run settings in code.

## Method overview

PRISM includes the following components:

1. **Modality encoding and sample-context modeling.** Independent modality encoders produce representations that are aggregated into a sample-level context.
2. **Training-bank-driven missing-aware retrieval.** Each modality retrieves from a reference bank constructed exclusively from fully observable training representations. An observed modality uses its current representation as the query; a missing modality uses a modality prototype.
3. **Modality enhancement and consistency regularization.** Modality-specific latent transformers integrate retrieved evidence. A distribution-level consistency loss regularizes the enhanced representations.
4. **Multi-branch sentiment prediction.** Original modality features, enhanced modality features, and sample context are combined for sentiment regression.
5. **Protocol-specific fusion adaptation.** After representation learning, the feature backbone, including the late Transformer, is frozen. Only the complete `fusion_all` prediction head is adapted for the selected fixed availability protocol.

The second training stage does not add a separate inference branch. Inference still uses the reference-bank retrieval mechanism.

## Environment

The reported experiments used Python 3.11.11 and PyTorch 2.5.1 with a CUDA 12.4 build on Linux. A CUDA-capable GPU is recommended for training. Shell-script examples assume Bash.

Install a PyTorch build appropriate for your CUDA environment first. The remaining core packages are:

```bash
python -m pip install transformers==4.30.2 numpy scipy scikit-learn einops tqdm
```

For the included tests, also install:

```bash
python -m pip install pytest
```

Keep `transformers==4.30.2`: the implementation imports BERT internals whose interfaces can change between versions. These installation commands list core dependencies, not a fully locked environment specification.

## Data and pretrained language model

### CMU-MOSI and CMU-MOSEI

Dataset information is available through the [CMU Multimodal SDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK). Follow the original datasets' access and usage conditions.

The training code expects preprocessed, aligned pickle files, not raw SDK downloads. Place compatible files in:

```text
datasets/
├── aligned_mosi.pkl
└── aligned_mosei.pkl
```

Each pickle must contain `train`, `dev`, and `test` splits. The loader expects examples in the form `((words, visual, acoustic), label, segment)`. Merely renaming a differently preprocessed dataset is not sufficient: feature dimensions, alignment, and split definitions must match the implementation. Only load pickle files from trusted sources.

### BERT

Download the tokenizer, configuration, and compatible model weights for [BERT-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) into **`BERT_EN/`** next to `main.py`.

```text
<extracted-source>/
├── main.py
├── IIE_model.py
├── modules/
├── BERT_EN/
│   ├── config.json
│   ├── vocab.txt
│   └── ... tokenizer files and compatible model weights
└── datasets/
    ├── aligned_mosi.pkl
    └── aligned_mosei.pkl
```

The directory name is `BERT_EN` with an underscore, as used by the current code.

## Fixed modality-availability protocols

The mask order is **text, audio, visual**. A value of 1 indicates an available modality; 0 indicates a missing modality. All samples within a fixed protocol use the same availability pattern.

| Protocol | Text | Audio | Visual | Available modalities |
| :---: | :---: | :---: | :---: | --- |
| 0 | 1 | 0 | 0 | Text |
| 1 | 0 | 1 | 0 | Audio |
| 2 | 0 | 0 | 1 | Visual |
| 3 | 1 | 1 | 0 | Text + audio |
| 4 | 1 | 0 | 1 | Text + visual |
| 5 | 0 | 1 | 1 | Audio + visual |
| 6 | 1 | 1 | 1 | Text + audio + visual |

## Training

### Method identifiers

The formal experiment runners use the following internal identifiers:

| Identifier | Configuration |
| --- | --- |
| `M0` | Legacy IIE retrieval baseline with mini-batch candidates |
| `M1` | Missing-aware retrieval from the training reference bank; representation-learning stage |
| `M2` | `M1` followed by fixed-protocol adaptation of the complete fusion head; retained PRISM configuration |
| `M3` | Additional adaptation after `M2`; historical ablation, not the retained final configuration |

`M2` requires a completed `M1` run for the same dataset, protocol, and seed. The runners resolve and verify the parent checkpoint automatically. Do not launch `M2` alone unless its matching `M1` artifacts already exist.

### Inspect a run plan

The following commands print the planned commands without training:

```bash
# One protocol and one seed: representation learning followed by fusion adaptation.
python run_e06_mosei_primary.py --protocols=0 --seeds=1111 --methods=M1,M2
python run_e07_mosi_external.py --protocols=0 --seeds=1111 --methods=M1,M2
```

### Train a small selection

After preparing the environment, datasets, and BERT files:

```bash
# CMU-MOSEI: two runs for protocol 0 and seed 1111.
E06_EXECUTE=1 python run_e06_mosei_primary.py \
  --protocols=0 --seeds=1111 --methods=M1,M2 --execute

# CMU-MOSI: the corresponding two runs.
E07_EXECUTE=1 python run_e07_mosi_external.py \
  --protocols=0 --seeds=1111 --methods=M1,M2 --execute
```

These are full training runs for a small selection, not one-epoch smoke tests.

### Train across protocols and seeds

The predefined seeds are `1111`, `2222`, `3407`, `5576`, and `7777`. Omitting `--protocols` and `--seeds` selects all seven protocols and all five seeds.

```bash
# Each command schedules 105 runs: 7 protocols × 5 seeds × 3 configurations.
E06_EXECUTE=1 python run_e06_mosei_primary.py --methods=M0,M1,M2 --execute
E07_EXECUTE=1 python run_e07_mosi_external.py --methods=M0,M1,M2 --execute
```

Always specify `--methods` explicitly: the MOSEI runner's historical default also includes the additional `M3` ablation. The separate `run_e08_protocol3_clean.py` provides the clean protocol-3 replication workflow; consult its `--help` and dry-run plan when reproducing that experiment.

The `E06`, `E07`, and similar prefixes are internal experiment identifiers, not model names. The generic `run.sh` and older shell scripts are not substitutes for the formal PRISM runners.

### Frozen primary settings

The primary runners use missing-aware training-bank retrieval with `retrieval_top_k=2`, `retrieval_temperature=0.1`, no cross-batch queue, and reference-bank refresh every epoch. Representation learning uses learning rate `2e-5`; fusion adaptation uses `1e-4`. Other settings, including batch size and representation dimension, depend on the protocol and are defined in the runners.

Later sensitivity experiments should be distinguished from these frozen primary settings; they do not retroactively change the configurations used for the main results.

## Evaluation and reproducibility

- Validation MAE controls checkpoint selection and early stopping. The formal workflow evaluates the selected checkpoint on the test set once per run.
- Adaptation starts from the matching representation-learning checkpoint and uses a fresh optimizer. The unadapted checkpoint remains a candidate if adaptation does not improve validation MAE.
- Reference banks contain training representations only. Validation and test samples are never retrieval candidates; missing placeholders are excluded, and training queries exclude their own sample.
- Reference banks are runtime resources rather than trainable checkpoint parameters. Rebuild them from the training split after loading the selected checkpoint.
- Each fixed protocol is trained and adapted independently. Run manifests record settings, data/code fingerprints, checkpoint lineage, and evaluation results.
- Existing completed runs can be reused after consistency checks. Ambiguous partial records or mismatched fingerprints are rejected rather than silently overwritten.

Generated artifacts use these directories:

```text
paper_experiments/manifests/   # Per-run JSON records
paper_experiments/registry.csv # Aggregated experiment registry
paper_checkpoints/            # Selected model checkpoints
paper_logs/                   # Training logs
paper_results/                # Audits and derived experiment outputs
```

To refresh the registry from available manifests:

```bash
python refresh_paper_registry.py --workspace .
```

Report the exact dataset preprocessing, protocol, seed, and selected configuration when comparing results. Binary sentiment metrics that include zero-valued labels must be distinguished from metrics that exclude them. For random-missing evaluation, distinguish nominal missing probability from realized missing rate after excluding all-missing samples.

## Additional experiments

The archive includes Python runners for component and retrieval ablations, adaptation-scope comparisons, retrieval-neighbor count, active bank size, retrieval temperature, fusion learning rate, and random-missing evaluation. Use each runner's `--help` and default dry-run mode to inspect its requirements before execution. Many depend on completed parent runs and their manifests; they are not independent quick-start scripts.

## Acknowledgments

This implementation builds on the IIE codebase associated with *Intra-Sample and Intra-Modal Enhancement for Multimodal Sentiment Analysis With Missing Modalities* by Zhuang et al. PRISM retains foundational components and introduces the training-bank retrieval and protocol-specific adaptation workflow described in its manuscript.

Additional source-level acknowledgments, including those for the latent Transformer and attention implementations, are retained in the corresponding files. Please acknowledge the relevant original work when reusing these components.

## Citation and licensing

The PRISM manuscript does not yet have publication metadata supplied with this release. A publication citation can be added when available; no journal acceptance or DOI is implied by this repository.

No standalone license file is included in the current source archive. Public availability alone does not grant unrestricted reuse rights. Check the applicable upstream terms and obtain clarification from the repository maintainers before redistribution or reuse beyond those terms.
