import metrics
from utils.data_loader import *
import threading
from concurrent.futures import ThreadPoolExecutor
import heapq
import torch
import pickle

cores = 8  

args = None
Ks = None
data_generator = None
USR_NUM, ITEM_NUM = None, None
N_TRAIN, N_TEST = None, None
BATCH_SIZE = None
_eval_cache_device = None
_test_items_padded = None
_test_item_lengths = None
_tail_item_indicator = None


def _ensure_eval_cache(device):
    global _eval_cache_device, _test_items_padded, _test_item_lengths, _tail_item_indicator

    device_key = str(device)
    if (_eval_cache_device == device_key and _test_items_padded is not None
            and _test_item_lengths is not None and _tail_item_indicator is not None):
        return

    max_test_len = max((len(items) for items in data_generator.test_set.values()), default=0)
    padded = np.full((USR_NUM, max_test_len), -1, dtype=np.int64)
    lengths = np.zeros(USR_NUM, dtype=np.int64)

    for user_id, items in data_generator.test_set.items():
        if len(items) == 0:
            continue
        item_array = np.asarray(items, dtype=np.int64)
        padded[user_id, :item_array.size] = item_array
        lengths[user_id] = item_array.size

    tail_indicator = np.zeros(ITEM_NUM, dtype=np.float64)
    if hasattr(data_generator, 'is_tail_item'):
        for item_id, is_tail in data_generator.is_tail_item.items():
            if 0 <= item_id < ITEM_NUM and is_tail == 1:
                tail_indicator[item_id] = 1.0
    _test_items_padded = torch.from_numpy(padded).to(device)
    _test_item_lengths = torch.from_numpy(lengths).to(device)
    _tail_item_indicator = torch.from_numpy(tail_indicator).to(device)
    _eval_cache_device = device_key


def _compute_topk_metrics_vectorized(topk_indices, user_batch_tensor):
    device = topk_indices.device
    _ensure_eval_cache(device)

    topk_indices = topk_indices.long()
    k_max = topk_indices.size(1)
    ks_tensor = torch.as_tensor(Ks, device=device, dtype=torch.long)
    k_indices = ks_tensor - 1

    batch_test_items = _test_items_padded[user_batch_tensor]
    valid_test_items = batch_test_items.ge(0)
    hit_matrix = ((topk_indices.unsqueeze(-1) == batch_test_items.unsqueeze(1))
                  & valid_test_items.unsqueeze(1)).any(dim=2).to(torch.float64)

    gt_lengths = _test_item_lengths[user_batch_tensor].to(torch.float64)
    positions = torch.arange(1, k_max + 1, device=device, dtype=torch.float64)
    discounts = 1.0 / torch.log2(torch.arange(2, k_max + 2, device=device, dtype=torch.float64))

    prefix_hits = torch.cumsum(hit_matrix, dim=1)
    precision_prefix = prefix_hits / positions.unsqueeze(0)
    ap_prefix = torch.cumsum(precision_prefix * hit_matrix, dim=1)
    dcg_prefix = torch.cumsum(hit_matrix * discounts.unsqueeze(0), dim=1)

    idcg_lookup = torch.zeros(k_max + 1, device=device, dtype=torch.float64)
    if k_max > 0:
        idcg_lookup[1:] = torch.cumsum(discounts, dim=0)

    hits_at_ks = prefix_hits.index_select(1, k_indices)
    recall = hits_at_ks / gt_lengths.unsqueeze(1)

    ap_at_ks = ap_prefix.index_select(1, k_indices)
    map_scores = torch.where(hits_at_ks > 0, ap_at_ks / hits_at_ks, torch.zeros_like(ap_at_ks))

    dcg_at_ks = dcg_prefix.index_select(1, k_indices)
    idcg_at_ks = idcg_lookup[hits_at_ks.long()]
    ndcg = torch.where(idcg_at_ks > 0, dcg_at_ks / idcg_at_ks, torch.zeros_like(dcg_at_ks))

    return recall, ndcg, map_scores


def ranklist_by_heapq(u, user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc, K_max_item_score

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]
    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(u, user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc


def get_performance(user_pos_test, r, auc, Ks, topk_items=None):
    recall, ndcg, map = [], [], []
    for K in Ks:
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        map.append(metrics.average_precision(r, K))
    return {
        'recall': np.array(recall),
        'ndcg': np.array(ndcg),
        'map': np.array(map),
        'topk_items': np.asarray(topk_items if topk_items is not None else [], dtype=np.int64)
    }
def test_one_user(x):
    rating = x[0]
    u = x[1]
    try:
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []
    user_pos_test = data_generator.test_set[u]

    all_items = set(range(ITEM_NUM))
    test_items = list(all_items - set(training_items))

    if args.test_flag == 'part':
        r, auc, rec_items = ranklist_by_heapq(u, user_pos_test, test_items, rating, Ks)
        topk_items = rec_items
    else:
        r, auc = ranklist_by_sorted(u, user_pos_test, test_items, rating, Ks)
        rec_items = None
        topk_items = np.asarray([], dtype=np.int64)

    return get_performance(user_pos_test, r, auc, Ks, topk_items)

def test_one_user_topk(x):
    
    topk_indices = x[0]  # TopK物品ID
    u = x[1]  # 用户ID

    user_pos_test = data_generator.test_set[u]

    K_max = max(Ks)
    K_max_item_score = topk_indices[:K_max]

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)

    auc = 0.

    return get_performance(user_pos_test, r, auc, Ks, K_max_item_score)

def test_batch_users(user_batch_data):

    results = []
    for x in user_batch_data:
        results.append(test_one_user(x))
    return results

def test_batch_users_topk(user_batch_data):

    results = []
    for x in user_batch_data:
        results.append(test_one_user_topk(x))
    return results


def test_torch(ua_embeddings, ia_embeddings, users_to_test, train_mask_gpu, drop_flag=False, batch_test_flag=False):

    if getattr(args, 'eval_mode', 'gpu') == 'cpu':
        result = {
            'recall': np.zeros(len(Ks)),
            'ndcg': np.zeros(len(Ks)),
            'map': np.zeros(len(Ks)),
            'tail_coverage': np.zeros(len(Ks)),
        }
        all_topk_items = []
        u_batch_size = BATCH_SIZE * 2
        n_test_users = len(users_to_test)
        n_user_batchs = n_test_users // u_batch_size + 1
        count = 0

        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size
            user_batch = users_to_test[start:end]
            if len(user_batch) == 0:
                continue

            item_batch = range(ITEM_NUM)
            u_g_embeddings = ua_embeddings[user_batch]
            i_g_embeddings = ia_embeddings[item_batch]
            rate_batch = torch.matmul(u_g_embeddings, torch.transpose(i_g_embeddings, 0, 1))
            rate_batch = rate_batch.detach().cpu().numpy()
            user_batch_rating_uid = list(zip(rate_batch, user_batch))

            chunk_size = max(1, len(user_batch_rating_uid) // max(1, (cores * 4)))
            user_chunks = [user_batch_rating_uid[i:i + chunk_size] for i in range(0, len(user_batch_rating_uid), chunk_size)]
            with ThreadPoolExecutor(max_workers=cores) as executor:
                chunk_results = list(executor.map(test_batch_users, user_chunks))
            batch_result = [result_item for chunk in chunk_results for result_item in chunk]

            count += len(batch_result)
            for re in batch_result:
                result['recall'] += re['recall'] / n_test_users
                result['ndcg'] += re['ndcg'] / n_test_users
                result['map'] += re['map'] / n_test_users
                if 'topk_items' in re and re['topk_items'].size > 0:
                    all_topk_items.append(re['topk_items'])

        assert count == n_test_users
        if all_topk_items and hasattr(data_generator, 'is_tail_item'):
            k_max = max(Ks)
            padded_topk_items = np.full((len(all_topk_items), k_max), -1, dtype=np.int64)
            for idx, items in enumerate(all_topk_items):
                valid_items = np.asarray(items, dtype=np.int64)[:k_max]
                padded_topk_items[idx, :len(valid_items)] = valid_items

            tail_item_indicator = np.zeros(ITEM_NUM, dtype=np.int64)
            for item_id in data_generator.is_tail_item:
                if 0 <= item_id < ITEM_NUM:
                    tail_item_indicator[item_id] = 1
            result['tail_coverage'] = metrics.tail_coverage_at_ks(
                padded_topk_items, tail_item_indicator, Ks
            )
        return result

    result = {
        'recall': np.zeros(len(Ks)),
        'ndcg': np.zeros(len(Ks)),
        'map': np.zeros(len(Ks)),
        'tail_coverage': np.zeros(len(Ks)),
    }

    n_test_users = len(users_to_test)
    device = ua_embeddings.device
    _ensure_eval_cache(device)
    test_users_tensor = torch.as_tensor(users_to_test, device=device, dtype=torch.long)
    item_embeddings = ia_embeddings
    k_max = max(Ks)
    all_topk_indices = []

    u_batch_size = BATCH_SIZE * 4
    n_user_batchs = n_test_users // u_batch_size + 1
    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = min((u_batch_id + 1) * u_batch_size, n_test_users)

        if start >= n_test_users:
            break

        user_batch_tensor = test_users_tensor[start:end]
        u_g_embeddings = ua_embeddings[user_batch_tensor]
        rate_batch = torch.matmul(u_g_embeddings, torch.transpose(item_embeddings, 0, 1))

        batch_train_mask = train_mask_gpu[user_batch_tensor]
        rate_batch[batch_train_mask] = float('-inf')

        topk_indices = torch.topk(rate_batch, k_max, dim=1).indices
        batch_recall, batch_ndcg, batch_map = _compute_topk_metrics_vectorized(topk_indices, user_batch_tensor)
        all_topk_indices.append(topk_indices)

        count += user_batch_tensor.shape[0]
        result['recall'] += batch_recall.sum(dim=0).cpu().numpy()
        result['ndcg'] += batch_ndcg.sum(dim=0).cpu().numpy()
        result['map'] += batch_map.sum(dim=0).cpu().numpy()

    assert count == n_test_users, f"Expected {n_test_users} users, got {count}"
    result['recall'] /= n_test_users
    result['ndcg'] /= n_test_users
    result['map'] /= n_test_users
    if all_topk_indices:
        all_topk_indices = torch.cat(all_topk_indices, dim=0)
        result['tail_coverage'] = metrics.tail_coverage_at_ks(all_topk_indices, _tail_item_indicator, Ks)
    return result
