# MMQF-DDI

**Multi-Scale Molecular Query Fusion for Drug–Drug Interaction Prediction**

MMQF-DDI predicts multi-class drug–drug interactions (DDIs) by combining global molecular semantics, local BRICS fragment representations, and relational graph information. Its main components are:

- Offline MolFormer molecular features and a densely connected semantic encoder.
- A trainable JK-GIN encoder for BRICS fragment graphs.
- Semantic-conditioned multi-query cross-attention for global–local fusion.
- A similarity-augmented drug graph and R-GCN for relational learning.
- Semantic–graph and fragment–graph contrastive objectives, together with a drug-pair classifier.

This README describes the code and data included in this directory. The commands below run the full model through `main.py`.

## 1. Environment setup

The supplied configuration targets Linux with Conda and an NVIDIA GPU. It specifies Python 3.8.20, PyTorch 2.4.0, CUDA runtime 12.4, PyTorch Geometric 2.5.3, and RDKit 2024.3.5. Use an NVIDIA driver compatible with the CUDA 12.4 runtime. A separate system CUDA toolkit is not required for the supplied PyTorch packages.

From the repository root, run:

```bash
conda env create -n MMQF -f environment_server.yaml
conda activate MMQF
```

The explicit `-n MMQF` overrides the environment name stored in the YAML file. Use the same name in any launcher scripts.

Verify the main dependencies:

```bash
python -c "import torch, torch_geometric, rdkit; print('PyTorch:', torch.__version__); print('PyG:', torch_geometric.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda)"
```

The YAML pins MKL and Intel OpenMP to avoid the `libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent` import error. Preserve these pins when recreating the environment.

## 2. Repository structure and data

```text
MMQF/
├── main.py                         # Training and evaluation entry point
├── model.py                        # Model definitions
├── Run_00.py                       # Local batch launcher; see Section 6
├── environment_server.yaml         # Conda environment
├── Deng/
│   ├── 0/
│   │   ├── 0/ ... 4/               # Five provided folds
│   │   │   ├── train.csv
│   │   │   └── test.csv
│   ├── ddi_class_65.csv
│   ├── drug_listxiao.csv
│   └── ...                         # Precomputed inputs listed below
└── Lin/
    ├── 0/
    │   ├── 0/ ... 4/
    │   │   ├── train.csv
    │   │   └── test.csv
    ├── DDI_1258.csv
    ├── drug_list_1258.csv
    └── ...
```

| Dataset | Drugs in the feature set | DDI classes |
|---|---:|---:|
| Deng | 572 | 65 |
| Lin | 1,258 | 100 |

Each split CSV contains `Drug1`, `Drug2`, and `Label`. Drug indices are zero-based and must match the row ordering of the molecular features, similarity matrices, and per-drug fragment inputs. Class labels range from `0` to `C - 1`.

`main.py` requires the following files inside each dataset directory:

| File | Purpose |
|---|---|
| `Molformer_smiles_features.pt` | Offline molecular features, with one row per drug |
| `molformer_sim_matrix.csv` | MolFormer cosine similarity matrix |
| `chem_Jacarrd_sim.csv` | Loaded by the current entry point; not used in the fused similarity score |
| `brics_weighted_similarity_matrix.npy` | IDF-weighted BRICS Jaccard similarity matrix |
| `brics_fragments_data.pkl` | Per-drug fragments and fragment IDF weights |
| `fragment_graph_data.pkl` | Fragment graphs and their exact vocabulary/index mapping |
| `drug_fragment_indices.pkl` | Per-drug indices into that fragment vocabulary |
| `<seed>/<fold>/train.csv` | Training drug pairs |
| `<seed>/<fold>/test.csv` | Test drug pairs |

The necessary precomputed inputs are included, so no preprocessing or MolFormer model download is required to run the supplied experiments. Keep these files together and preserve their indexing. `ChemBERTa_smiles_features.pt` and `brics_drug_list.pkl`, where present, are not loaded by the current training entry point.

## 3. Run one fold

Run all commands from the repository root because the code uses relative paths. Create the output directory first:

```bash
mkdir -p result

# Optional: limit CPU threads per training process.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
```

### Deng

```bash
python main.py \
  --dataset Deng --seed 0 --fold 0 --gpu_id 0 \
  --model Model_F1_C_S2_new --rels 65 \
  --n_epochs 400 --batch_size 2048 --dropout 0.5 \
  --graph_alpha 0.6 --min_sim 0.6 \
  --temperature 0.2 --cl_alpha 0.7 --w_cl 0.1 --lr 0.002 \
  --filename result/Deng_seed0.csv
```

### Lin

```bash
python main.py \
  --dataset Lin --seed 0 --fold 0 --gpu_id 0 \
  --model Model_F1_C_S2_new --rels 100 \
  --n_epochs 400 --batch_size 8192 --dropout 0.3 \
  --graph_alpha 0.7 --min_sim 0.7 \
  --temperature 0.05 --cl_alpha 0.7 --w_cl 0.1 --lr 0.001 \
  --filename result/Lin_seed0.csv
```

`Model_F1_C_S2_new` selects the full model with the trainable JK-GIN fragment encoder. Do not substitute `Model_F1_C_S2` when reproducing this configuration: its fragment encoder implementation differs.

`--gpu_id` selects the GPU index visible to PyTorch. If CUDA is unavailable, the code falls back to CPU, which may be substantially slower. For a one-epoch execution check, replace `--n_epochs 400` with `--n_epochs 1` and use a separate output CSV.

## 4. Run the provided five folds

The packaged splits are under `Deng/0/` and `Lin/0/`. The following Bash loops run folds 0–4 sequentially, using the configurations above.

### Deng

```bash
for fold in 0 1 2 3 4; do
  python main.py \
    --dataset Deng --seed 0 --fold "$fold" --gpu_id 0 \
    --model Model_F1_C_S2_new --rels 65 \
    --n_epochs 400 --batch_size 2048 --dropout 0.5 \
    --graph_alpha 0.6 --min_sim 0.6 \
    --temperature 0.2 --cl_alpha 0.7 --w_cl 0.1 --lr 0.002 \
    --filename result/Deng_seed0.csv || break
done
```

### Lin

```bash
for fold in 0 1 2 3 4; do
  python main.py \
    --dataset Lin --seed 0 --fold "$fold" --gpu_id 0 \
    --model Model_F1_C_S2_new --rels 100 \
    --n_epochs 400 --batch_size 8192 --dropout 0.3 \
    --graph_alpha 0.7 --min_sim 0.7 \
    --temperature 0.05 --cl_alpha 0.7 --w_cl 0.1 --lr 0.001 \
    --filename result/Lin_seed0.csv || break
done
```

## 5. Outputs and evaluation

The program prints training loss and accuracy, then evaluates the test split at epochs **1, 300, 350, and 400**. A one-epoch run is also evaluated. Each evaluation appends one row to the CSV specified by `--filename`.

The CSV records the model, split identifier, fold, hyperparameters, and six metrics:

- ACC: overall classification accuracy.
- AUPR: micro-averaged area under the precision–recall curve, computed by trapezoidal integration.
- AUROC: micro-averaged ROC AUC.
- F1, Precision, and Recall: macro-averaged classification metrics.

The CSV column `n_epochs` stores the evaluated epoch for that row. For final-epoch five-fold reporting, select the five rows with `n_epochs == 400`, one per fold, and compute their mean and standard deviation separately for each metric. Rerunning a command appends more rows to an existing CSV; use a new filename to keep independent runs separate.

The current entry point does not save model checkpoints or prediction NPZ files and does not automatically aggregate fold results. Its console label `Val F1` refers to the supplied **test** split; there is no separate validation split or early stopping in this entry point. Use the predetermined final epoch for reporting rather than selecting the best test epoch.

### Current reproducibility details

- `--seed` selects the data directory; the current `main.py` does not initialize Python, NumPy, or PyTorch random seeds. The provided splits are fixed, but training results can vary across runs.
- Training DDI graph edges are made bidirectional with the same relation type. The classification training set itself is not augmented with reversed pairs.
- Auxiliary similarity is `graph_alpha * MolFormer_similarity + (1 - graph_alpha) * BRICS_similarity`. Edges meeting `min_sim` are retained after self-loop and overlap filtering. Similarity values are not used as continuous message-passing weights.
- The current overlap filter uses both training and test pair identities, without test labels. Thus graph construction depends on the supplied test pair list. A protocol requiring completely training-only graph construction must change this filter and rerun the experiments; it is not the behavior of this snapshot.
- The contrastive objective uses all drug nodes in the supplied dataset. The packaged entry point is not a strict inductive cold-start implementation.

## 6. Using `Run_00.py`

The included `Run_00.py` retains machine-specific settings from a local experiment. As packaged, it points to an external `main_cold1.py`, requests `Model_WO_RGCN`, and selects GPU 3. It is therefore not a ready-to-run launcher for the full model in this directory.

To use it for these experiments, update:

1. `conda_path`: the absolute path returned by `which conda` on your machine.
2. `script_path`: the absolute path to this repository's `main.py`.
3. `env_name`: `MMQF`, matching the environment created above.
4. `gpu`: an available GPU index, usually `0` for a single-GPU setup.
5. The `--model` argument: `Model_F1_C_S2_new`.
6. The output filename: a dataset-specific reproduction CSV rather than the retained cold-start filename.

The active loop currently uses the Deng configuration. To run Lin, use the Lin configuration consistently, including `rels=100`, batch size, learning rate, dropout, and similarity/contrastive parameters. Alternatively, use the explicit commands in Sections 3–4 without modifying the launcher.

## 7. Troubleshooting

- **Missing dataset or feature file:** run from the repository root and check the exact filenames in Section 2. Only split identifier `0` is packaged here.
- **Missing output directory:** create `result/` before running; `main.py` does not create parent directories for the output CSV.
- **Invalid GPU index:** set `--gpu_id` to an index visible to the process. With `CUDA_VISIBLE_DEVICES` set, GPU indices are relative to the visible device list.
- **CUDA out of memory:** reduce `--batch_size`. Full-graph and fragment computations also consume memory, so reducing batch size may not reduce all memory use.
- **`iJIT_NotifyEvent` import error:** recreate the environment with the supplied MKL/OpenMP pins and check that a later package installation has not replaced them.
- **Different results across repeated runs:** fixed splits do not imply fixed model initialization; see the random-seed limitation above.

## 8. Citation and data distribution

Bibliographic details will be added when the paper is available. Before redistributing dataset files or pretrained assets, check the terms of their original providers and include the appropriate source acknowledgments and permissions. This README does not grant a separate license for third-party data.
