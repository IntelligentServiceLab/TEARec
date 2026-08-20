import hashlib
import json
import os

import torch


def load_description_texts(dataset_dir, n_users):
    description_path = os.path.join(dataset_dir, 'descriptions.jsonl')
    if not os.path.exists(description_path):
        return None

    descriptions = [''] * n_users
    with open(description_path, 'r', encoding='utf-8') as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for key, value in record.items():
                index = int(key)
                if 0 <= index < n_users:
                    descriptions[index] = value.strip()

    return descriptions


def build_text_embedding_cache_path(cache_dir, model_name_or_path, max_length):
    cache_key = hashlib.md5(
        f'{model_name_or_path}|{max_length}'.encode('utf-8')
    ).hexdigest()[:12]
    return os.path.join(cache_dir, f'app_text_embeddings_{cache_key}.pt')


def mean_pool_embeddings(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    pooled = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return pooled / denom


def encode_texts_with_bert(
    texts,
    model_name_or_path,
    batch_size=32,
    max_length=256,
    device=None,
    cache_path=None,
    model_cache_dir=None,
    local_files_only=True,
):
    if cache_path and os.path.exists(cache_path):
        return torch.load(cache_path, map_location='cpu').float()

    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=model_cache_dir,
        local_files_only=local_files_only,
    )
    model = AutoModel.from_pretrained(
        model_name_or_path,
        cache_dir=model_cache_dir,
        local_files_only=local_files_only,
    ).to(device)
    model.eval()

    all_embeddings = []
    normalized_texts = [text if text else 'no description available' for text in texts]

    with torch.no_grad():
        for start in range(0, len(normalized_texts), batch_size):
            batch_texts = normalized_texts[start:start + batch_size]
            encoded_inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt',
            )
            encoded_inputs = {
                key: value.to(device)
                for key, value in encoded_inputs.items()
            }
            model_outputs = model(**encoded_inputs)
            batch_embeddings = mean_pool_embeddings(
                model_outputs.last_hidden_state,
                encoded_inputs['attention_mask'],
            )
            all_embeddings.append(batch_embeddings.cpu())

    embeddings = torch.cat(all_embeddings, dim=0).float()
    if cache_path:
        torch.save(embeddings, cache_path)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embeddings


def _squared_l2_distance(x, centers):
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    c_norm = (centers ** 2).sum(dim=1).unsqueeze(0)
    dist = x_norm + c_norm - 2 * torch.matmul(x, centers.t())
    return dist.clamp_min_(0.0)


def _kmeans_pp_init(x, k, generator=None):
    n = x.size(0)
    device = x.device
    first_idx = torch.randint(n, (1,), device=device, generator=generator)
    centers = [x[first_idx].squeeze(0)]

    closest_dist = _squared_l2_distance(x, centers[0].unsqueeze(0)).squeeze(1)
    for _ in range(1, k):
        probs = closest_dist / closest_dist.sum().clamp_min(1e-12)
        next_idx = torch.multinomial(probs, 1, generator=generator)
        next_center = x[next_idx].squeeze(0)
        centers.append(next_center)
        new_dist = _squared_l2_distance(x, next_center.unsqueeze(0)).squeeze(1)
        closest_dist = torch.minimum(closest_dist, new_dist)
    return torch.stack(centers, dim=0)


def _run_gpu_kmeans_once(x, k, max_iter=30, tol=1e-4, seed=2025):
    device = x.device
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    centers = _kmeans_pp_init(x, k, generator=g)
    prev_labels = None

    for _ in range(max_iter):
        dist = _squared_l2_distance(x, centers)
        labels = torch.argmin(dist, dim=1)

        if prev_labels is not None and torch.equal(labels, prev_labels):
            break
        prev_labels = labels

        new_centers = torch.zeros_like(centers)
        counts = torch.bincount(labels, minlength=k).float().unsqueeze(1)
        new_centers.scatter_add_(0, labels.unsqueeze(1).expand(-1, x.size(1)), x)

        empty = (counts.squeeze(1) == 0)
        counts = counts.clamp_min(1.0)
        new_centers = new_centers / counts

        if empty.any():
            farthest_idx = torch.topk(dist.min(dim=1).values, int(empty.sum().item()), largest=True).indices
            new_centers[empty] = x[farthest_idx]

        shift = torch.norm(new_centers - centers, dim=1).mean()
        centers = new_centers
        if shift <= tol:
            break

    final_dist = _squared_l2_distance(x, centers)
    final_labels = torch.argmin(final_dist, dim=1)
    inertia = final_dist[torch.arange(x.size(0), device=device), final_labels].sum()
    return centers, final_labels, inertia


def gpu_kmeans(x, k, max_iter=30, tol=1e-4, n_init=3, seed=2025):
    """Pure PyTorch GPU KMeans using squared Euclidean distance."""
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    x = x.float()
    if not x.is_cuda and torch.cuda.is_available():
        x = x.cuda()

    best_centers, best_labels, best_inertia = None, None, None
    for init_idx in range(n_init):
        centers, labels, inertia = _run_gpu_kmeans_once(
            x, k, max_iter=max_iter, tol=tol, seed=seed + init_idx)
        if best_inertia is None or inertia < best_inertia:
            best_centers = centers.clone()
            best_labels = labels.clone()
            best_inertia = inertia.clone()

    return best_centers, best_labels


def apply_pca(x, out_dim, center=True, q=None, niter=2):
    """Apply PCA with torch.pca_lowrank and return projected features."""
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    x = x.float()
    if out_dim <= 0 or out_dim >= x.size(1):
        return x

    if center:
        x_mean = x.mean(dim=0, keepdim=True)
        x_centered = x - x_mean
    else:
        x_centered = x

    if q is None:
        q = min(max(out_dim + 8, out_dim), x.size(1))
    q = min(q, x.size(1))

    U, S, V = torch.pca_lowrank(x_centered, q=q, center=False, niter=niter)
    V = V[:, :out_dim]
    return torch.matmul(x_centered, V)
