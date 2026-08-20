import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F
import numpy as np

class LGCN(nn.Module):
    """双图对比学习"""
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_layers = len(weight_size)
        self.dropout_list = nn.ModuleList()

        # 共享初始embedding
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self._init_weight_()

    def _init_weight_(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj_u, adj_i, adj_interact):
        """
        双图传播前向过程
        Args:
            adj_u: 同质用户图 [n_users, n_users]
            adj_i: 同质物品图 [n_items, n_items]
            adj_interact: 异构交互图 [n_users+n_items, n_users+n_items]

        Returns:
            homo_u, homo_i: 同质图embedding
            hetero_u, hetero_i: 异构图embedding
        """
        # ========== 同构图传播 ==========
        hu = self.user_embedding.weight
        hi = self.item_embedding.weight
        embedding_u = [hu]
        embedding_i = [hi]

        for i in range(self.n_layers):
            hu = torch.sparse.mm(adj_u, embedding_u[-1])
            hi = torch.sparse.mm(adj_i, embedding_i[-1])
            embedding_u.append(hu)
            embedding_i.append(hi)


        u_emb_stack = torch.stack(embedding_u, dim=1)
        i_emb_stack = torch.stack(embedding_i, dim=1)
        homo_u = torch.mean(u_emb_stack, dim=1, keepdim=False)
        homo_i = torch.mean(i_emb_stack, dim=1, keepdim=False)

        # ========== 异构交互图传播 ==========
        ego_embeddings = torch.cat([self.user_embedding.weight,
                                    self.item_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]

        for i in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(adj_interact, ego_embeddings)
            all_embeddings.append(ego_embeddings)


        all_emb_stack = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_emb_stack, dim=1, keepdim=False)
        hetero_u = all_embeddings[:self.n_users]
        hetero_i = all_embeddings[self.n_users:]

        return homo_u, homo_i, hetero_u, hetero_i
