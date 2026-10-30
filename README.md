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
├── Run_00.py                       # Local batch launcher;
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
| `brics_weighted_similarity_matrix.npy` | IDF-weighted BRICS Jaccard similarity matrix |
| `brics_fragments_data.pkl` | Per-drug fragments and fragment IDF weights |
| `fragment_graph_data.pkl` | Fragment graphs and their exact vocabulary/index mapping |
| `drug_fragment_indices.pkl` | Per-drug indices into that fragment vocabulary |
| `<seed>/<fold>/train.csv` | Training drug pairs |
| `<seed>/<fold>/test.csv` | Test drug pairs |

The necessary precomputed inputs are included, so no preprocessing or MolFormer model download is required to run the supplied experiments. Keep these files together and preserve their indexing. 

