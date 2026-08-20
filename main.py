import numpy as np
import torch
import torch.nn as nn
import torch.sparse as sparse  
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import datetime
import math
from models import *
from utils.helper import *
from utils.batch_test import *
from utils.parser import parse_args
from utils.data_loader import Data
from utils.loss_utils import (bpr_loss, contrastive_loss,
                               ssl_loss as compute_ssl_loss,
                               sampled_softmax_loss, proto_cl_loss,
                               anchored_cross_view_bpr_loss)
from utils.sparse_mat_interface import sparse_mx_to_torch_sparse_tensor
from utils.helper import set_random_seed

args = parse_args()
if args.loss_type == 'softmax':
    args.loss_type = 'ssm'
if args.homo_only and args.hetero_only:
    raise ValueError("--homo_only and --hetero_only cannot be enabled at the same time.")

print("=" * 80)
print("运行命令:")
print(f"python {' '.join(sys.argv)}")
print("=" * 80)
print()

set_random_seed(args.seed)

data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)

USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size
Ks = eval(args.Ks)

import utils.batch_test as batch_test_module
batch_test_module.args = args
batch_test_module.Ks = Ks
batch_test_module.data_generator = data_generator  
batch_test_module.USR_NUM, batch_test_module.ITEM_NUM = USR_NUM, ITEM_NUM
batch_test_module.N_TRAIN, batch_test_module.N_TEST = N_TRAIN, N_TEST
batch_test_module.BATCH_SIZE = BATCH_SIZE

def lamb(epoch):
    epoch += 0
    return 0.95 ** (epoch / 14)

result = []
alpha11=args.alpha1
alpha12=args.alpha2
Lambda=args.Lambda
type=f"3_4_{args.model_name}_lambda_{Lambda}"
name=f"{type}_lr_{args.lr}__alpha_{alpha11}_ssl_reg_{args.ssl_reg}_dataset_{args.dataset}_emb_size_{args.embed_size}_layers_{args.layer_size}"

os.makedirs("logs", exist_ok=True)

writer = None
test_writers = {}
if args.tensorboard:
    import datetime
    timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    base_tb_log_dir = f"runs/{timestamp}-{name}"

    os.makedirs(base_tb_log_dir, exist_ok=True)
    writer = SummaryWriter(base_tb_log_dir)

    k_list_str = ','.join(map(str, Ks))
    for metric in ['Recall', 'NDCG', 'MAP', 'TailCoverage']:
        for k in Ks:
            test_log_dir = f"{base_tb_log_dir}/Test/{metric}@[{k_list_str}]/{k}"
            os.makedirs(test_log_dir, exist_ok=True)
            test_writers[f'{metric}@{k}'] = SummaryWriter(test_log_dir)

    print(f"TensorBoard日志已启用: {base_tb_log_dir}")
    print(f"测试指标分离日志已创建，共{len(test_writers)}个writer")
else:
    print("TensorBoard日志已关闭")

txt = open(f"logs/{name}.txt", "a")
txt.write(name + "\n")

def quan_jac(new_matrix, matrix, chunk_size=2048):
    if torch.cuda.is_available():
        n = new_matrix.shape[0]
        matrix_t = torch.from_numpy(np.ascontiguousarray(matrix, dtype=np.float32)).cuda()
        new_matrix_t = torch.from_numpy(np.ascontiguousarray(new_matrix, dtype=np.float32)).cuda()

        D = torch.sum(new_matrix_t * matrix_t, dim=1)
        del new_matrix_t
        torch.cuda.empty_cache()

        sim = np.zeros((n, n), dtype=np.float32)
        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            chunk_t = torch.from_numpy(np.ascontiguousarray(new_matrix[s:e], dtype=np.float32)).cuda()
            intersection = torch.mm(chunk_t, matrix_t.T)
            denom = D[s:e].unsqueeze(1) + D.unsqueeze(0) - intersection
            sim[s:e] = (intersection / denom).cpu().numpy()
            del chunk_t, intersection, denom
            torch.cuda.empty_cache()

        del matrix_t, D
        torch.cuda.empty_cache()
        return sim
    else:
        intersection = np.dot(new_matrix, matrix.T)
        D = np.diag(intersection)
        U = D[:, None] + D.T
        fenmu = U - intersection
        return intersection / fenmu

def new_inter_matrix(inter):
    if torch.cuda.is_available():
        t = torch.from_numpy(np.ascontiguousarray(inter, dtype=np.float32)).cuda()
        item_pop = 1.0 / torch.log(t.sum(dim=0) + Lambda)
        user_pop = 1.0 / torch.log(t.sum(dim=1) + Lambda)
        new_user = (t * item_pop.unsqueeze(0)).cpu().numpy()
        new_item = (t.T * user_pop.unsqueeze(0)).cpu().numpy()
        del t
        torch.cuda.empty_cache()
        return new_user, new_item
    else:
        item_sum = np.sum(inter, axis=0) + Lambda
        user_sum = np.sum(inter, axis=1) + Lambda
        item_pop = 1.0 / np.log(item_sum)
        user_pop = 1.0 / np.log(user_sum)
        new_user_inter = inter * item_pop
        new_item_inter = inter.T * user_pop
        return new_user_inter, new_item_inter

class Model_Wrapper(object):
    def __init__(self, data_config):
        self.mess_dropout = eval(args.mess_dropout)

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = [args.embed_size] * args.layer_size
        self.n_layers = len(self.weight_size)

        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.verbose = args.verbose

        print(f'Model: {args.model_name} (LightGCN backbone, {self.n_layers} layers)')

        self.weights_save_path = f"w/{args.dataset}"
        """
        *********************************************************
        Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
        """

        self.ssl_sim_power = args.ssl_sim_power
        self.ssl_coeff_power = args.ssl_coeff_power

        self.model = LGCN(self.n_users, self.n_items, self.emb_dim, self.weight_size, self.mess_dropout)

        self.adj_user_norm, self.adj_item_norm, self.norm_interact_adj = self.build_adj(args.data_path + args.dataset + '/train.txt')

        self.model = self.model.cuda()
        self.cl_temp = args.cl_temp

        alpha_init = min(max(args.alpha_init, 1e-4), 1 - 1e-4)
        alpha_logit_init = math.log(alpha_init / (1 - alpha_init))
        self.alpha_logit = nn.Parameter(torch.tensor(alpha_logit_init, dtype=torch.float32, device='cuda'))
        self.homo_scale_log = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device='cuda'))
        self.hetero_scale_log = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device='cuda'))

        if args.cl_reg > 0 and (not args.homo_only) and (not args.hetero_only):
            self._build_cl_degree_masks(args.cl_degree_pct)

        optim_params = list(self.model.parameters())
        optim_params.extend([self.alpha_logit, self.homo_scale_log, self.hetero_scale_log])
        self.optimizer = optim.Adam(optim_params, lr=self.lr)

        self.lr_scheduler = self.set_lr_scheduler()

        self.app2cluster = None
        if args.proto_reg > 0:
            self._init_text_clusters()

        self.users_to_test = list(data_generator.test_set.keys())
        print(f"[OK] Cached {len(self.users_to_test)} test users")

        self.train_mask_gpu = self._build_train_mask_gpu()
        print(f"[OK] Built train mask on GPU: {self.train_mask_gpu.shape}, sparsity={self.train_mask_gpu.sum().item() / self.train_mask_gpu.numel():.4f}")


    def _build_cl_degree_masks(self, top_pct):
        """构建 CL 度数筛选 mask：只对度数前 top_pct 的 app/tpl 做对比学习"""
        R = data_generator.R.tocsr()
        user_deg = np.array(R.sum(axis=1)).flatten()   # [n_users]
        item_deg = np.array(R.sum(axis=0)).flatten()   # [n_items]

        u_thresh = np.percentile(user_deg, 100 * (1 - top_pct))
        i_thresh = np.percentile(item_deg, 100 * (1 - top_pct))

        self.cl_user_mask = torch.from_numpy(user_deg >= u_thresh).cuda()  # bool [n_users]
        self.cl_item_mask = torch.from_numpy(item_deg >= i_thresh).cuda()  # bool [n_items]

        n_u = self.cl_user_mask.sum().item()
        n_i = self.cl_item_mask.sum().item()
        print(f"[CL] Degree filter: top {top_pct*100:.0f}% -> "
              f"app {n_u}/{self.n_users} (thresh>={u_thresh:.0f}), "
              f"tpl {n_i}/{self.n_items} (thresh>={i_thresh:.0f})")

    def _init_text_clusters(self):
        """加载文本embedding → KMeans聚类 → 得到静态 app2cluster 映射"""
        from utils.text_cluster_utils import (load_description_texts,
                                              encode_texts_with_bert,
                                              build_text_embedding_cache_path,
                                              gpu_kmeans,
                                              apply_pca)

        dataset_dir = args.data_path + args.dataset
        cache_dir = data_generator.cache_dir
        project_root = os.path.dirname(os.path.abspath(__file__))
        model_cache_dir = os.path.join(project_root, 'cache')

        semantic_npy_path = os.path.join(dataset_dir, 'semantic_signature_embeddings.npy')
        if os.path.exists(semantic_npy_path):
            text_emb = torch.from_numpy(np.load(semantic_npy_path)).float()
            if text_emb.shape[0] != self.n_users:
                print(f"[Proto CL] semantic_signature_embeddings.npy row count {text_emb.shape[0]} != n_users {self.n_users}, disabled.")
                self.app2cluster = None
                return
            print(f"[Proto CL] Using existing semantic vectors: {text_emb.shape}")
        else:
            descriptions = load_description_texts(dataset_dir, self.n_users)
            if descriptions is None:
                print(f"[Proto CL] descriptions.jsonl not found in {dataset_dir}, disabled.")
                self.app2cluster = None
                return

            emb_cache_path = build_text_embedding_cache_path(
                cache_dir, args.bert_model, args.bert_max_length)
            text_emb = encode_texts_with_bert(
                descriptions, args.bert_model,
                batch_size=args.bert_batch_size,
                max_length=args.bert_max_length,
                cache_path=emb_cache_path,
                model_cache_dir=model_cache_dir)
            print(f"[Proto CL] Encoded texts with {args.bert_model}: {text_emb.shape}")

        if args.cluster_mode == 'pca':
            pca_dim = min(args.cluster_pca_dim, text_emb.shape[1])
            text_emb = apply_pca(text_emb, pca_dim)
            print(f"[Proto CL] Applied PCA before clustering: {text_emb.shape}")
        elif args.cluster_mode != 'raw':
            raise ValueError(f"Unknown cluster_mode: {args.cluster_mode}")

        k = min(args.n_clusters, self.n_users)
        cluster_cache = os.path.join(cache_dir, f'{args.cluster_mode}_text_kmeans_k{k}.pt')
        if os.path.exists(cluster_cache):
            labels = torch.load(cluster_cache, map_location='cpu')
            print(f"[Proto CL] Loaded cached cluster labels: k={k}")
        else:
            print(f"[Proto CL] Running GPU KMeans (k={k})...")
            _, labels = gpu_kmeans(
                text_emb, k=k, max_iter=30, tol=1e-4, n_init=3, seed=args.seed)
            labels = labels.cpu().long()
            torch.save(labels, cluster_cache)
            print(f"[Proto CL] GPU KMeans done, saved to {cluster_cache}")

        self.app2cluster = labels.cuda()
        self.n_clusters = k
        cluster_sizes = torch.bincount(self.app2cluster, minlength=k)
        print(f"[Proto CL] Cluster sizes: min={cluster_sizes.min().item()}, "
              f"max={cluster_sizes.max().item()}, mean={cluster_sizes.float().mean().item():.1f}")

    def compute_proto_centroids(self, fused_user_emb):
        """用当前epoch的fused user embedding计算每个cluster的中心"""
        K = self.n_clusters
        dim = fused_user_emb.shape[1]
        centroids = torch.zeros(K, dim, device='cuda')
        counts = torch.bincount(self.app2cluster, minlength=K).float().unsqueeze(1)
        centroids.scatter_add_(0, self.app2cluster.unsqueeze(1).expand(-1, dim), fused_user_emb)
        centroids = centroids / counts.clamp_min(1)
        return centroids

    def get_fusion_alpha(self):
        logits = torch.stack([self.alpha_logit, -self.alpha_logit])
        weights = torch.softmax(logits, dim=0)
        return weights[0], weights[1]

    def fuse_embeddings(self, homo_u, homo_i, hetero_u, hetero_i):
        if args.homo_only:
            return homo_u, homo_i
        if args.hetero_only:
            return hetero_u, hetero_i

        alpha_h, alpha_g = self.get_fusion_alpha()
        homo_u = F.normalize(homo_u, dim=1) * torch.exp(self.homo_scale_log)
        homo_i = F.normalize(homo_i, dim=1) * torch.exp(self.homo_scale_log)
        hetero_u = F.normalize(hetero_u, dim=1) * torch.exp(self.hetero_scale_log)
        hetero_i = F.normalize(hetero_i, dim=1) * torch.exp(self.hetero_scale_log)

        ua = alpha_h * homo_u + alpha_g * hetero_u
        ia = alpha_h * homo_i + alpha_g * hetero_i
        return ua, ia

    def _sample_ssm_neg_items(self, users, pos_items, neg_count):
        """从 k-hop 之外为每个App采样若干负样本 item。"""
        valid_mask = ~self.user_item_khop[users]  # [B, n_items]
        valid_mask.scatter_(1, pos_items.unsqueeze(1), False)
        valid_counts = valid_mask.sum(dim=1)
        k = int(min(neg_count, int(valid_counts.min().item())))
        if k <= 0:
            raise RuntimeError('No valid negatives available for SSM sampling.')

        rand_scores = torch.rand(valid_mask.shape, device=valid_mask.device)
        rand_scores = rand_scores.masked_fill(~valid_mask, -1.0)
        neg_items = torch.topk(rand_scores, k=k, dim=1).indices  # [B, k]
        return neg_items

    def _build_train_mask_gpu(self):

        print("Building train mask on GPU...")
        train_mask = torch.zeros(self.n_users, self.n_items, dtype=torch.bool)

        user_indices = []
        item_indices = []

        for u, items in data_generator.train_items.items():
            if len(items) > 0:
                user_indices.extend([u] * len(items))
                item_indices.extend(items)

        if len(user_indices) > 0:
            user_indices = torch.LongTensor(user_indices)
            item_indices = torch.LongTensor(item_indices)
            train_mask[user_indices, item_indices] = True

        return train_mask.cuda()

    def build_adj(self, file):
        cache_dir = data_generator.cache_dir
        jaccard_cache = os.path.join(cache_dir, f"{Lambda}_jaccard_matrices.npz")
        dataset_dir = os.path.dirname(file)
        legacy_jaccard_cache = os.path.join(dataset_dir, f"{Lambda}_jaccard_matrices.npz")
        jaccard_cache_to_load = jaccard_cache if os.path.exists(jaccard_cache) else legacy_jaccard_cache

        if os.path.exists(jaccard_cache_to_load):
            print(f"[Cache] 加载 Jaccard 相似度矩阵缓存: {jaccard_cache_to_load}")
            cache_data = np.load(jaccard_cache_to_load, allow_pickle=True)
            J_u = cache_data['J_u']
            J_i = cache_data['J_i']
            user_inter = cache_data['user_inter']
            print(f"[Cache] Jaccard 矩阵加载完成: J_u {J_u.shape}, J_i {J_i.shape}")
        else:
            print(f"[Cache] 缓存不存在，开始构建 Jaccard 相似度矩阵...")
            user_inter = np.zeros((USR_NUM, ITEM_NUM))  # 用户与物品的交互矩阵
            items_inter = np.zeros((ITEM_NUM, USR_NUM))  # 物品与用户的交互矩阵
            with open(file) as f:
                for l in f.readlines():
                    if len(l) == 0: break
                    l = l.strip("\n").split(" ")
                    uid = int(l[0])
                    items = [int(j) for j in l[1:]]
                    user_inter[uid, items] = 1
                    items_inter[items, uid] = 1

            # 对交互矩阵去偏处理（降低热门物品/活跃用户权重）
            new_user_inter, new_item_inter = new_inter_matrix(user_inter)

            print("[Cache] 计算用户 Jaccard 相似度...")
            J_u = quan_jac(new_user_inter, user_inter)
            print("[Cache] 计算物品 Jaccard 相似度...")
            J_i = quan_jac(new_item_inter, items_inter)

            print(f"[Cache] 保存 Jaccard 矩阵缓存: {jaccard_cache}")
            np.savez(jaccard_cache, J_u=J_u, J_i=J_i, user_inter=user_inter)
            print(f"[Cache] Jaccard 矩阵缓存保存完成")

        J_u_t = torch.from_numpy(np.ascontiguousarray(J_u, dtype=np.float32)).cuda()
        reg1_u = -(J_u_t - alpha11)
        reg1_u.fill_diagonal_(0.0)
        self.coeff_u = torch.relu(reg1_u)
        self.coeff_uc = self.coeff_u                # 给对比学习
        self.coeff_u = torch.pow(self.coeff_u, self.ssl_coeff_power)
        self.coeff_u = args.ssl_reg * self.coeff_u
        mask_u = J_u_t > alpha11
        mask_u.fill_diagonal_(False)
        row_u, col_u = mask_u.nonzero(as_tuple=True)
        vals_u = J_u_t[row_u, col_u]
        self.adj_user = torch.sparse_coo_tensor(
            torch.stack([row_u, col_u]), vals_u, (USR_NUM, USR_NUM)).coalesce()
        deg_u = torch.sparse.sum(self.adj_user, dim=1).to_dense()
        deg_u_inv = torch.where(deg_u > 0, 1.0 / deg_u, torch.zeros_like(deg_u))
        self.adj_user_norm = torch.sparse_coo_tensor(
            torch.stack([row_u, col_u]), vals_u * deg_u_inv[row_u],
            (USR_NUM, USR_NUM)).coalesce()
        del reg1_u, mask_u

        J_i_t = torch.from_numpy(np.ascontiguousarray(J_i, dtype=np.float32)).cuda()

        reg1_i = -(J_i_t - alpha12)
        reg1_i.fill_diagonal_(0.0)
        self.coeff_i = torch.relu(reg1_i)
        self.coeff_ic = self.coeff_i
        self.coeff_i = torch.pow(self.coeff_i, self.ssl_coeff_power)
        self.coeff_i = args.ssl_reg * self.coeff_i
        mask_i = J_i_t > alpha12
        mask_i.fill_diagonal_(False)
        row_i, col_i = mask_i.nonzero(as_tuple=True)
        vals_i = J_i_t[row_i, col_i]
        self.adj_item = torch.sparse_coo_tensor(
            torch.stack([row_i, col_i]), vals_i, (ITEM_NUM, ITEM_NUM)).coalesce()
        deg_i = torch.sparse.sum(self.adj_item, dim=1).to_dense()
        deg_i_inv = torch.where(deg_i > 0, 1.0 / deg_i, torch.zeros_like(deg_i))
        self.adj_item_norm = torch.sparse_coo_tensor(
            torch.stack([row_i, col_i]), vals_i * deg_i_inv[row_i],
            (ITEM_NUM, ITEM_NUM)).coalesce()
        del reg1_i, mask_i, J_u_t, J_i_t

        # Build k-hop reachability only for sampled softmax.
        self.user_item_khop = None
        if args.loss_type == 'ssm':
            khop = args.khop
            R_sp = sp.csr_matrix(user_inter)  # [n_users, n_items]
            if khop <= 1:
                reachable = R_sp
            else:
                I = R_sp.T.dot(R_sp)  # [n_items, n_items] item-item 共现
                I_bin = (I > 0).astype(np.float32)
                I_power = I_bin.copy()
                for _ in range((khop - 1) // 2 - 1):
                    I_power = I_power.dot(I_bin)
                    I_power = (I_power > 0).astype(np.float32)
                reachable = R_sp.dot(I_power)
                reachable = reachable + R_sp
                reachable = (reachable > 0).astype(np.float32)
            self.user_item_khop = torch.from_numpy(reachable.toarray()).bool().cuda()
            print(f"[k-hop] khop={khop}, reachable nnz={reachable.nnz} "
                  f"(density={reachable.nnz / (USR_NUM * ITEM_NUM) * 100:.2f}%), "
                  f"GPU tensor: {self.user_item_khop.shape}")

        norm_interact_cache = os.path.join(cache_dir, "norm_interact_adj.npz")
        legacy_norm_interact_cache = os.path.join(dataset_dir, "norm_interact_adj.npz")
        norm_interact_cache_to_load = norm_interact_cache if os.path.exists(norm_interact_cache) else legacy_norm_interact_cache

        if os.path.exists(norm_interact_cache_to_load):
            print(f"[Cache] 加载归一化交互图缓存: {norm_interact_cache_to_load}")
            cache_data = np.load(norm_interact_cache_to_load, allow_pickle=True)
            data = cache_data['data']
            row = cache_data['row']
            col = cache_data['col']
            shape = tuple(cache_data['shape'])
            indices = torch.from_numpy(np.vstack((row, col)).astype(np.int64))
            values = torch.from_numpy(data.astype(np.float32))
            self.norm_interact_adj = torch.sparse_coo_tensor(
                indices, values, torch.Size(shape), dtype=torch.float32).coalesce().cuda()
            n_nodes = shape[0]
            print(f"[Cache] 归一化交互图加载完成: {n_nodes} nodes, {len(data)} edges")
        else:
            print("Building heterogeneous interaction graph...")
            n_nodes = self.n_users + self.n_items
            interact_r = data_generator.R.tocsr().astype(np.float32)
            zero_user = sp.csr_matrix((self.n_users, self.n_users), dtype=np.float32)
            zero_item = sp.csr_matrix((self.n_items, self.n_items), dtype=np.float32)
            interact_adj = sp.bmat(
                [[zero_user, interact_r], [interact_r.T, zero_item]],
                format='csr', dtype=np.float32)

            interact_adj = interact_adj.tocoo()
            rowsum = np.array(interact_adj.sum(1)).flatten()
            d_inv_sqrt = np.power(rowsum, -0.5)
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            norm_interact_adj = d_mat_inv_sqrt.dot(interact_adj).dot(d_mat_inv_sqrt).tocoo()

            self.norm_interact_adj = sparse_mx_to_torch_sparse_tensor(norm_interact_adj).cuda()
            print(f"Interaction graph built: {n_nodes} nodes, {interact_adj.nnz} edges")

            print(f"[Cache] 保存归一化交互图缓存: {norm_interact_cache}")
            np.savez(norm_interact_cache,
                     data=norm_interact_adj.data,
                     row=norm_interact_adj.row,
                     col=norm_interact_adj.col,
                     shape=norm_interact_adj.shape)
            print(f"[Cache] 归一化交互图缓存保存完成")

        if args.enable_tail_app_aug and data_generator.tail_items:
            R_coo = data_generator.R.tocoo()
            user_idx_np = R_coo.row.astype(np.int64)
            item_idx_np = R_coo.col.astype(np.int64)

            u_idx = torch.from_numpy(user_idx_np).cuda()
            i_idx = torch.from_numpy(item_idx_np).cuda()
            i_node_idx = i_idx + self.n_users

            self._orig_row = torch.cat([u_idx, i_node_idx])
            self._orig_col = torch.cat([i_node_idx, u_idx])
            self._orig_val = torch.ones(len(self._orig_row), device='cuda')
            self._tail_items = torch.tensor(sorted(data_generator.tail_items), device='cuda', dtype=torch.long)
            self._n_nodes = self.n_users + self.n_items
            self._base_norm_interact_adj = self.norm_interact_adj

            # 原始交互对（用于计算每个app对已使用tpl的平均打分阈值）
            self._raw_interact_user = u_idx
            self._raw_interact_item = i_idx
            self._user_interact_count = torch.bincount(u_idx, minlength=self.n_users).float()

            # 记录 tail tpl 是否已被某个 app 使用，避免重复加边
            tail_list = sorted(data_generator.tail_items)
            tail_pos = {item: idx for idx, item in enumerate(tail_list)}
            user_tail_exists = np.zeros((self.n_users, len(tail_list)), dtype=np.bool_)
            for u, i in zip(user_idx_np, item_idx_np):
                idx = tail_pos.get(i)
                if idx is not None:
                    user_tail_exists[u, idx] = True
            self._user_tail_exists = torch.from_numpy(user_tail_exists).cuda()

            print(f"[Aug] Rule-based tpl->app augmentation enabled: {len(self._tail_items)} tail tpl, lambda={args.aug_lambda}")

        return self.adj_user_norm, self.adj_item_norm, self.norm_interact_adj

    def rebuild_aug_interact_graph(self, user_embeddings, item_embeddings):
        """每epoch基于当前embedding重建增强交互图：原始u-i边 + 满足阈值规则的 tail tpl- app 边"""
        tail_emb = item_embeddings[self._tail_items]  # [n_tail, dim]

        # 每个 app 对已使用 tpl 的平均打分阈值：mean_j <e_u, e_j>
        interact_item_emb = item_embeddings[self._raw_interact_item]  # [n_interacts, dim]
        interact_user_emb = user_embeddings[self._raw_interact_user]   # [n_interacts, dim]
        interact_scores = torch.sum(interact_user_emb * interact_item_emb, dim=1)  # [n_interacts]
        user_score_sum = torch.zeros(self.n_users, device='cuda')
        user_score_sum.scatter_add_(0, self._raw_interact_user, interact_scores)
        user_score_mean = user_score_sum / self._user_interact_count.clamp_min(1.0)

        # 候选 tail tpl 对所有 app 的点积得分 [n_users, n_tail]
        score_mat = torch.matmul(user_embeddings, tail_emb.T)

        # tail tpl 与 app 的点积 > app 对已使用 tpl 的平均点积
        mask = score_mat > user_score_mean.unsqueeze(1)
        # 排除已存在的 app-tail 交互
        mask = mask & (~self._user_tail_exists)

        aug_user_idx, aug_tail_pos = mask.nonzero(as_tuple=True)
        if len(aug_user_idx) == 0:
            self.norm_interact_adj = self._base_norm_interact_adj
            return 0

        aug_item_idx = self._tail_items[aug_tail_pos] + self.n_users
        aug_row = torch.cat([aug_user_idx, aug_item_idx])
        aug_col = torch.cat([aug_item_idx, aug_user_idx])
        aug_val = torch.full((len(aug_row),), args.aug_lambda, device='cuda')

        all_row = torch.cat([self._orig_row, aug_row])
        all_col = torch.cat([self._orig_col, aug_col])
        all_val = torch.cat([self._orig_val, aug_val])

        adj = torch.sparse_coo_tensor(
            torch.stack([all_row, all_col]), all_val,
            (self._n_nodes, self._n_nodes)).coalesce()
        deg = torch.sparse.sum(adj, dim=1).to_dense()
        deg_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
        idx = adj.indices()
        vals = adj.values() * deg_inv_sqrt[idx[0]] * deg_inv_sqrt[idx[1]]
        self.norm_interact_adj = torch.sparse_coo_tensor(
            idx, vals, (self._n_nodes, self._n_nodes)).coalesce()
        return len(aug_user_idx)

    def set_lr_scheduler(self):  
        fac = lamb
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        return scheduler

    def save_model(self):
        ensureDir(self.weights_save_path)
        torch.save({
            'model': self.model.state_dict(),
            'alpha_logit': self.alpha_logit.detach(),
            'homo_scale_log': self.homo_scale_log.detach(),
            'hetero_scale_log': self.hetero_scale_log.detach(),
        }, self.weights_save_path)

    def load_model(self):
        checkpoint = torch.load(self.weights_save_path, map_location='cuda')
        self.model.load_state_dict(checkpoint['model'])
        with torch.no_grad():
            self.alpha_logit.copy_(checkpoint['alpha_logit'])
            self.homo_scale_log.copy_(checkpoint['homo_scale_log'])
            self.hetero_scale_log.copy_(checkpoint['hetero_scale_log'])

    def test(self, users_to_test, drop_flag=False, batch_test_flag=False):
        self.model.eval()  # 评估模式，batchnorm和Drop层不起作用
        with torch.no_grad():
            homo_u, homo_i, hetero_u, hetero_i = self.model(
                self.adj_user_norm, self.adj_item_norm, self.norm_interact_adj)

            ua_embeddings, ia_embeddings = self.fuse_embeddings(homo_u, homo_i, hetero_u, hetero_i)
            if not hasattr(self, '_alpha_logged'):
                alpha_h, alpha_g = self.get_fusion_alpha()
                print(f"[Fusion] weights: homo={alpha_h.item():.4f}, hetero={alpha_g.item():.4f}")
                print(f"[Fusion] scales: homo={torch.exp(self.homo_scale_log).item():.4f}, hetero={torch.exp(self.hetero_scale_log).item():.4f}")
                self._alpha_logged = True

        result = test_torch(ua_embeddings, ia_embeddings, users_to_test, self.train_mask_gpu)

        return result

    def train(self):
        training_time_list = []
        loss_loger, rec_loger, ndcg_loger, map_loger, cov_loger = [], [], [], [], []
        stopping_step = 10
        should_stop = False
        cur_best_pre_0 = 0.

        n_batch = data_generator.n_train // args.batch_size + 1

        for epoch in range(args.epoch):
            t1 = time()
            loss_tensor = torch.tensor(0.0, device='cuda')
            sup_loss_tensor = torch.tensor(0.0, device='cuda')
            reg_loss_tensor = torch.tensor(0.0, device='cuda')
            cl_loss_tensor = torch.tensor(0.0, device='cuda')
            ssl_loss_tensor = torch.tensor(0.0, device='cuda')
            n_batch = data_generator.n_train // args.batch_size + 1

            enable_cl = (args.cl_reg > 0 and hasattr(self, 'cl_user_mask') and not args.hetero_only)
            enable_ssl = args.ssl_reg > 0 and not args.hetero_only
            enable_aug = args.enable_tail_app_aug and hasattr(self, '_orig_row') and not args.homo_only
            enable_proto = args.proto_reg > 0 and self.app2cluster is not None and not args.hetero_only

            if enable_aug:
                with torch.no_grad():
                    _, _, hetero_u, hetero_i = self.model(
                        self.adj_user_norm, self.adj_item_norm, self.norm_interact_adj)
                n_aug = self.rebuild_aug_interact_graph(hetero_u, hetero_i)
                if epoch == 0:
                    print(f"[Aug] Epoch {epoch}: added {n_aug} tail-tpl<->app edges (bidirectional)")

            if enable_proto:
                with torch.no_grad():
                    homo_u_pre, homo_i_pre, hetero_u_pre, hetero_i_pre = self.model(
                        self.adj_user_norm, self.adj_item_norm, self.norm_interact_adj)
                    fused_u, _ = self.fuse_embeddings(homo_u_pre, homo_i_pre, hetero_u_pre, hetero_i_pre)
                proto_centroids = self.compute_proto_centroids(fused_u.detach())
                del homo_u_pre, homo_i_pre, hetero_u_pre, hetero_i_pre, fused_u

            if args.loss_type == 'ssm':
                epoch_users_list, epoch_pos_list = [], []

                for _ in range(n_batch):
                    users, pos_items, _ = data_generator.sample()
                    epoch_users_list.append(users)
                    epoch_pos_list.append(pos_items)

                epoch_users = torch.LongTensor(np.concatenate(epoch_users_list)).cuda()
                epoch_pos_items = torch.LongTensor(np.concatenate(epoch_pos_list)).cuda()

                del epoch_users_list, epoch_pos_list
            else:
                epoch_users_list, epoch_pos_list, epoch_neg_list = [], [], []

                for _ in range(n_batch):
                    users, pos_items, neg_items = data_generator.sample()
                    epoch_users_list.append(users)
                    epoch_pos_list.append(pos_items)
                    epoch_neg_list.append(neg_items)

                epoch_users = torch.LongTensor(np.concatenate(epoch_users_list)).cuda()
                epoch_pos_items = torch.LongTensor(np.concatenate(epoch_pos_list)).cuda()
                epoch_neg_items = torch.LongTensor(np.concatenate(epoch_neg_list)).cuda()

                del epoch_users_list, epoch_pos_list, epoch_neg_list

            for idx in range(n_batch):
                self.model.train()  # 模型为训练模式，像Dropout，Normalize这些层就会起作用，测试模式不会

                self.optimizer.zero_grad()

                start = idx * args.batch_size
                end = (idx + 1) * args.batch_size
                
                batch_cl_loss = torch.tensor(0.0, device='cuda')
                batch_ssl_loss = torch.tensor(0.0, device='cuda')
                batch_proto_loss = torch.tensor(0.0, device='cuda')
                batch_cross_loss = torch.tensor(0.0, device='cuda')

                
                
                
                users = epoch_users[start:end]
                pos_items = epoch_pos_items[start:end]

                homo_u, homo_i, hetero_u, hetero_i = self.model(
                    self.adj_user_norm, self.adj_item_norm, self.norm_interact_adj)

                ua_embeddings, ia_embeddings = self.fuse_embeddings(homo_u, homo_i, hetero_u, hetero_i)

                u_g_embeddings = ua_embeddings[users]
                pos_i_g_embeddings = ia_embeddings[pos_items]

                if args.loss_type == 'ssm':
                    neg_items = self._sample_ssm_neg_items(users, pos_items, args.ssm_neg_count)
                    neg_i_g_embeddings = ia_embeddings[neg_items]
                    batch_sup_loss, batch_reg_loss = sampled_softmax_loss(
                        u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings,
                        args.softmax_temp, self.decay, self.batch_size)
                else:
                    neg_items = epoch_neg_items[start:end]
                    neg_i_g_embeddings = ia_embeddings[neg_items]
                    batch_sup_loss, batch_reg_loss = bpr_loss(
                        u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings,
                        self.decay, self.batch_size)

                if self.alpha_logit is not None and idx == 0 and epoch % 10 == 0:
                    alpha_h, alpha_g = self.get_fusion_alpha()
                    msg = f"[Fusion] Epoch {epoch}: homo={alpha_h.item():.4f}, hetero={alpha_g.item():.4f}"
                    msg += f", scale_h={torch.exp(self.homo_scale_log).item():.4f}, scale_g={torch.exp(self.hetero_scale_log).item():.4f}"
                    print(msg)

                if enable_cl:
                    cl_loss_u = torch.tensor(0.0, device='cuda')
                    cl_loss_i = torch.tensor(0.0, device='cuda')

                    cl_u = users[self.cl_user_mask[users]]
                    if len(cl_u) > 1:
                        coeff_u_batch = self.coeff_uc[cl_u][:, cl_u]
                        cl_loss_u = contrastive_loss(
                            homo_u[cl_u], hetero_u[cl_u], coeff_u_batch, self.cl_temp)

                    unique_pos_items = torch.unique(pos_items)
                    cl_i = unique_pos_items[self.cl_item_mask[unique_pos_items]]
                    if len(cl_i) > 1:
                        coeff_i_batch = self.coeff_ic[cl_i][:, cl_i]
                        cl_loss_i = contrastive_loss(
                            homo_i[cl_i], hetero_i[cl_i], coeff_i_batch, self.cl_temp)

                    batch_cl_loss = cl_loss_u + cl_loss_i

                if enable_ssl:
                    batch_ssl_loss = compute_ssl_loss(
                        homo_u, homo_i, users, pos_items,
                        self.coeff_u, self.coeff_i, self.ssl_sim_power)
                if enable_proto:
                    batch_proto_loss = proto_cl_loss(
                        ua_embeddings, users, proto_centroids,
                        self.app2cluster, args.proto_temp)

                batch_loss = batch_sup_loss + batch_reg_loss + args.cl_reg * batch_cl_loss + batch_ssl_loss + args.proto_reg * batch_proto_loss + args.cross_reg * batch_cross_loss
                batch_loss.backward()
                self.optimizer.step()

                loss_tensor += batch_loss.detach()
                sup_loss_tensor += batch_sup_loss.detach()
                reg_loss_tensor += batch_reg_loss.detach()
                cl_loss_tensor += (args.cl_reg * batch_cl_loss + args.proto_reg * batch_proto_loss + args.cross_reg * batch_cross_loss).detach()
                ssl_loss_tensor += batch_ssl_loss.detach()

            self.lr_scheduler.step()  

            if args.loss_type == 'ssm':
                del epoch_users, epoch_pos_items
            else:
                del epoch_users, epoch_pos_items, epoch_neg_items

            loss = loss_tensor.item()
            sup_loss = sup_loss_tensor.item()
            reg_loss = reg_loss_tensor.item()
            cl_loss = cl_loss_tensor.item()
            ssl_loss = ssl_loss_tensor.item()

            if args.loss_type == 'ssm':
                del ua_embeddings, ia_embeddings, homo_u, homo_i, hetero_u, hetero_i
            else:
                del ua_embeddings, ia_embeddings, neg_i_g_embeddings, homo_u, homo_i, hetero_u, hetero_i


            if math.isnan(loss) == True:
                print('ERROR: loss is nan.')
                sys.exit()

            if (epoch + 1) % 10 != 0:
                if args.verbose > 0 and epoch % args.verbose == 0:
                    perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f +%.5f +%.5f +%.5f], loss=%s' % (
                        epoch, time() - t1, loss, sup_loss, reg_loss, cl_loss, ssl_loss, args.loss_type)
                    print(perf_str)

                    if writer:
                        writer.add_scalar('Loss/Total', loss, epoch)
                        writer.add_scalar('Loss/Supervised', sup_loss, epoch)
                        writer.add_scalar('Loss/Regularization', reg_loss, epoch)
                        writer.add_scalar('Loss/CL', cl_loss, epoch)
                        writer.add_scalar('Loss/SSL', ssl_loss, epoch)

            t2 = time()
            if epoch % 10 and epoch < 200:
                continue
            if epoch > 200 and epoch % 5:
                continue
            ret = self.test(self.users_to_test, drop_flag=True)
            training_time_list.append(t2 - t1)

            t3 = time()

            recall = ret['recall']
            ndcg = ret['ndcg']
            Map = ret['map']
            TailCoverage = ret['tail_coverage']
            recall = [round(num, 5) for num in recall]
            ndcg = [round(num, 5) for num in ndcg]
            Map = [round(num, 5) for num in Map]
            TailCoverage = [round(num, 5) for num in TailCoverage]

            loss_loger.append(loss)
            rec_loger.append(recall)
            ndcg_loger.append(ndcg)
            map_loger.append(Map)
            cov_loger.append(TailCoverage)

            tt1 = t2 - t1
            tt2 = t3 - t2

            if args.verbose > 0:
                perf_str = f'Epoch {epoch} [{tt1:.1f}s + {tt2:.1f}s]: train==[{loss:.3f}={sup_loss:.3f} +{reg_loss:.3f} +{cl_loss:.4f} +{ssl_loss:.4f}], loss={args.loss_type},' \
                           f' recall={recall}, ndcg={ndcg}, map={Map}, tail_cov={TailCoverage}'
                result.append(perf_str + "\n")

                global txt
                txt.write(perf_str + "\n")
                txt.close()
                txt = open(f"logs/{name}.txt", "a")
                print(perf_str)

                if writer:
                    for i, k in enumerate(Ks):
                        if f'Recall@{k}' in test_writers:
                            test_writers[f'Recall@{k}'].add_scalar('Recall', recall[i], epoch)
                        if f'NDCG@{k}' in test_writers:
                            test_writers[f'NDCG@{k}'].add_scalar('NDCG', ndcg[i], epoch)
                        if f'MAP@{k}' in test_writers:
                            test_writers[f'MAP@{k}'].add_scalar('MAP', Map[i], epoch)
                        if f'TailCoverage@{k}' in test_writers:
                            test_writers[f'TailCoverage@{k}'].add_scalar('TailCoverage', TailCoverage[i], epoch)
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['ndcg'][-1], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=8)

            if should_stop:
                break

            if ret['ndcg'][-1] == cur_best_pre_0 and args.save_flag == 1:
                self.save_model()
                print('save the weights in path: ', self.weights_save_path)

        if args.save_recom:
            results_save_path = r'./output/%s/rec_result.csv' % (args.dataset)
            self.save_recResult(results_save_path)

        if rec_loger != []:
            self.print_final_results(rec_loger, ndcg_loger, map_loger, cov_loger)

    def save_recResult(self, outputPath):
        recommendResult = {}
        u_batch_size = BATCH_SIZE * 2
        i_batch_size = BATCH_SIZE

        n_test_users = len(self.users_to_test)
        n_user_batchs = n_test_users // u_batch_size + 1
        count = 0

        self.model.eval()
        with torch.no_grad():
            homo_u, homo_i, hetero_u, hetero_i = self.model(
                self.adj_user_norm, self.adj_item_norm, self.norm_interact_adj)

            ua_embeddings, ia_embeddings = self.fuse_embeddings(homo_u, homo_i, hetero_u, hetero_i)

        embeddings=torch.cat((ua_embeddings, ia_embeddings),dim=0)
        torch.save(embeddings, 'embeddings.pt')
        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size
            user_batch = self.users_to_test[start: end]
            item_batch = range(ITEM_NUM)
            u_g_embeddings = ua_embeddings[user_batch]
            i_g_embeddings = ia_embeddings[item_batch]
            rate_batch = torch.matmul(u_g_embeddings, torch.transpose(i_g_embeddings, 0, 1))
            rate_batch = rate_batch.detach().cpu().numpy()
            user_rating_uid = zip(rate_batch, user_batch)
            for rating, u in user_rating_uid:
                training_items = data_generator.train_items[u]
                user_pos_test = data_generator.test_set[u]
                all_items = set(range(ITEM_NUM))
                test_items = list(all_items - set(training_items))
                item_score = {}
                for i in test_items:
                    item_score[i] = rating[i]
                K_max = max(Ks)
                K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)
                recommendResult[u] = K_max_item_score

        ensureDir(outputPath)
        with open(outputPath, 'w') as f:
            print("----the recommend result has %s items." % (len(recommendResult)))
            for key in recommendResult.keys():
                outString = ""
                for v in recommendResult[key]:
                    outString = outString + "," + str(v)
                f.write("%s%s\n" % (key, outString))

    def print_final_results(self, rec_loger, ndcg_loger, map_loger, cov_loger):
        recs = np.array(rec_loger)
        best_rec_0 = max(recs[:, 0])
        idx = list(recs[:, 0]).index(best_rec_0)
        t = time() - t0

        final_perf = f"Best Iter=[{idx}]@[{t:.1f}]\trecall={rec_loger[idx]}, ndcg={ndcg_loger[idx]}, map={map_loger[idx]}, tail_cov={cov_loger[idx]}"
        txt.write(final_perf + "\n")
        txt.close()
        print(final_perf)

if __name__ == '__main__':

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    # """
    plain_adj, norm_adj, _ = data_generator.get_adj_mat()

    config['norm_adj'] = norm_adj

    t0 = time()

    print("ok")
    Engine = Model_Wrapper(data_config=config)
    if args.pretrain:
        print('pretrain path: ', Engine.weights_save_path)
        if os.path.exists(Engine.weights_save_path):
            Engine.load_model()
            ret = Engine.test(Engine.users_to_test, drop_flag=True)
            cur_best_pre_0 = ret['recall'][0]

            pretrain_ret = 'pretrained model recall=[%.5f, %.5f],' \
                           'ndcg=[%.5f, %.5f], map=[%.5f, %.5f]' % \
                           (ret['recall'][0], ret['recall'][-1],
                            ret['ndcg'][0], ret['ndcg'][-1],
                            ret['map'][0], ret['map'][-1])
            print(pretrain_ret)
        else:
            print('Cannot load pretrained model. Start training from stratch')
    else:
        print('without pretraining')
    Engine.train()

    # 关闭TensorBoard writer
    if writer:
        writer.close()
        # 关闭所有测试指标的专用writer
        for key, test_writer in test_writers.items():
            test_writer.close()
        print(f"TensorBoard日志已关闭，共关闭{len(test_writers) + 1}个writer")
