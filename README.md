# TEARec

**Tail-Aware Edge Augmentation and Semantic Prototype Supervision for Graph-Based Third-Party Library Recommendation**

TEARec is a graph-based third-party library recommendation framework for multi-view collaborative filtering. 

## Main Components

- Jaccard-based App and TPL similarity graphs with popularity debiasing
- Heterogeneous App-TPL interaction graph
- Tail-aware edge augmentation for message propagation
- Branch-Norm fusion of homogeneous and heterogeneous graph views
- Contrastive learning between graph views
- Similarity-supervised learning
- Text-based semantic prototype supervision
- Sampled Softmax and standard BPR objectives

- Recall, NDCG, MAP, and Tail Coverage metrics

## Environment

Tested on:

- Python 3.11.13
- PyTorch 2.8.0+cu128
- CUDA 12.8
- NumPy 1.26.4
- SciPy 1.16.0
- scikit-learn 1.7.1
- GPU: NVIDIA RTX 5070, 12 GB VRAM



## Installation

```bash
pip install -r requirements.txt
```

### Optional C++ Extension for Faster Sampling

The C++ sampler is recommended for large datasets because it significantly reduces negative-sampling overhead.

```bash
pip install pybind11 cppimport
```

The extension in `source/sample.cpp` is compiled automatically on first use. If compilation fails, the code falls back to the Python sampler.

## Usage

Run the command from the project root:

```bash
python main.py
```

For a short startup smoke test:

```bash
python main.py --epoch 1
```

The default configuration uses **MLO-3** and the full TEARec configuration. The first run may build and cache similarity matrices and the normalized interaction graph.

View all command-line options with:

```bash
python main.py --help
```

## Default Configuration

The current defaults are:

| Parameter | Default |
|---|---:|
| `dataset` | `MLO-3` |
| `embed_size` | `128` |
| `layer_size` | `3` |
| `batch_size` | `4096` |
| `epoch` | `800` |
| `lr` | `1e-4` |
| `Lambda` | `8` |
| `alpha1` | `0.53` |
| `alpha2` | `0.48` |
| `loss_type` | `ssm` |
| `softmax_temp` | `0.5` |
| `ssm_neg_count` | `64` |
| `cl_reg` | `0.004` |
| `cl_temp` | `0.15` |
| `cl_degree_pct` | `0.2` |
| `ssl_reg` | `1e-4` |
| `proto_reg` | `0.001` |
| `proto_temp` | `0.1` |
| `n_clusters` | `256` |
| `cluster_mode` | `pca` |
| `cluster_pca_dim` | `128` |
| `enable_tail_app_aug` | `1` |
| `aug_lambda` | `2.0` |
| `Ks` | `[5, 10]` |
| `seed` | `2026` |
| `tensorboard` | `0` |
| `eval_mode` | `gpu` |



## Dataset Format

Place datasets under:

```text
datasets/<dataset_name>/
```

Each interaction file contains one user per line:

```text
user_id item_id_1 item_id_2 item_id_3 ...
```

Required files:

```text
datasets/<dataset_name>/
├── train.txt
├── test.txt
├── head.txt
└── tail.txt
```

Optional files used by semantic prototype supervision:

```text
datasets/<dataset_name>/
├── descriptions.jsonl
├── semantic_signature_embeddings.npy
├── semantic_signature_embeddings.jsonl
└── semantic_signature_embeddings_meta.json
```

The repository includes two datasets:

| Public name | Directory name |
|---|---|
| **MLO-3** | `MLO-3` |
| **MLO-5** | `MLO-5` |

Use the dataset directory name with `--dataset`, for example:

```bash
python main.py --dataset MLO-3
python main.py --dataset MLO-5
```

The corresponding directories are:

```text
datasets/MLO-3/
datasets/MLO-5/
```

`head.txt` and `tail.txt` contain one library ID per line. They are used for tail-aware graph augmentation and Tail Coverage evaluation; 

### Dataset Source

The dataset is derived from the [**MALib-Dataset**](https://github.com/malibdata/MALib-Dataset). 

## Text Embeddings and Semantic Prototypes

Semantic prototype supervision is enabled when:

```text
proto_reg > 0
```

The pipeline first looks for a precomputed embedding file:

```text
datasets/<dataset_name>/semantic_signature_embeddings.npy
```

If the file exists, it is loaded directly. This is the recommended path for reproducing the reported experiments.

The provided metadata records the embedding source and shape. For the current datasets, the provided vectors were generated with:

```text
BAAI/bge-large-en-v1.5
```

and have dimension:

```text
1024
```

If the `.npy` file is not available, the code can encode `descriptions.jsonl` with a locally available Hugging Face Transformers model. The default model is `BAAI/bge-large-en-v1.5`, matching the provided vectors. Install the dependency:

```bash
pip install transformers
```

The text encoding implementation is in:

```text
utils/text_cluster_utils.py
```

Important requirements for re-encoding:

- The tokenizer and model must be available in the local Transformers cache.
- `local_files_only=True` is used by default, so the model is not downloaded automatically at runtime.
- The number of embedding rows must equal the number of users/apps.
- Row `i` must correspond to app ID `i`.
- To reproduce the provided vectors, use the same embedding model and preprocessing configuration.

After loading or generating text vectors, the code optionally applies PCA and GPU KMeans clustering before prototype supervision.

## Training and Evaluation Outputs

Depending on the enabled options, the program writes:

```text
logs/                         Training and evaluation logs
runs/                         TensorBoard event files
w/<dataset>                   Saved model checkpoint
output/<dataset>/             Saved recommendation results
embeddings.pt                 Saved fused user/item embeddings
```

Dataset-specific graph and clustering caches are stored under:

```text
datasets/<dataset_name>/cache/
```

Typical cached files include:

```text
s_adj_mat.npz
s_norm_adj_mat.npz
s_mean_adj_mat.npz
<Lambda>_jaccard_matrices.npz
norm_interact_adj.npz
pca_text_kmeans_k<k>.pt
```

## Evaluation Metrics

The default evaluation reports:

- Recall@K
- NDCG@K
- MAP@K
- Tail Coverage@K

The default values are:

```text
K = 5, 10
```

Evaluation implementations:

```bash
python main.py --eval_mode gpu
python main.py --eval_mode cpu
```



