import torch
import torch.nn.functional as F


def sampled_softmax_loss(user_emb, pos_item_emb, neg_item_emb,
                         temperature, decay, batch_size):
    """
    SSM / sampled softmax loss with explicitly sampled negatives.

    Args:
        user_emb: [batch, dim]
        pos_item_emb: [batch, dim]
        neg_item_emb: [batch, n_neg, dim]
        temperature: softmax 温度
        decay: 正则化系数
        batch_size: batch 大小
    """
    pos_scores = torch.sum(user_emb * pos_item_emb, dim=1, keepdim=True) / temperature  # [B,1]
    neg_scores = torch.sum(user_emb.unsqueeze(1) * neg_item_emb, dim=2) / temperature   # [B,M]
    logits = torch.cat([pos_scores, neg_scores], dim=1)
    log_denom = torch.logsumexp(logits, dim=1)
    loss = -(pos_scores.squeeze(1) - log_denom)

    regularizer = (0.5 * (user_emb ** 2).sum() +
                   0.5 * (pos_item_emb ** 2).sum() +
                   0.5 * (neg_item_emb ** 2).sum() / max(1, neg_item_emb.size(1))) / batch_size
    reg_loss = decay * regularizer

    return loss.mean() / 10.0, reg_loss

def bpr_loss(users, pos_items, neg_items, decay, batch_size):
    pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
    neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

    regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
    regularizer = regularizer / batch_size

    mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
    emb_loss = decay * regularizer
    return mf_loss, emb_loss



def contrastive_loss(homo_emb, hetero_emb, coeff_matrix, temperature=0.2):
    batch_size = homo_emb.shape[0]
    homo_emb = F.normalize(homo_emb, dim=1)
    hetero_emb = F.normalize(hetero_emb, dim=1)

    sim_e2h = torch.matmul(hetero_emb, homo_emb.T) / temperature

    diagonal_mask = torch.eye(batch_size, device=coeff_matrix.device)
    weights = coeff_matrix * (1 - diagonal_mask)

    exp_e2h = torch.exp(sim_e2h)
    exp_pos = torch.diagonal(exp_e2h)
    weighted_neg_sum = torch.sum(weights * exp_e2h, dim=1)
    denominator = exp_pos + weighted_neg_sum

    pos_sim = torch.diagonal(sim_e2h)
    loss = -torch.mean(pos_sim - torch.log(denominator + 1e-8))
    return loss


def ssl_loss(homo_u, homo_i, users, pos_items, coeff_u, coeff_i, ssl_sim_power):
    n_homo_u = F.normalize(homo_u, dim=1)
    n_homo_i = F.normalize(homo_i, dim=1)

    sim_user = torch.pow(
        torch.relu(torch.matmul(n_homo_u[users], torch.transpose(n_homo_u, 0, 1))),
        ssl_sim_power
    )
    sim_item = torch.pow(
        torch.relu(torch.matmul(n_homo_i[pos_items], torch.transpose(n_homo_i, 0, 1))),
        ssl_sim_power
    )

    cos_loss_u = torch.sum(torch.mul(coeff_u[users], sim_user), dim=0)
    cos_loss_i = torch.sum(torch.mul(coeff_i[pos_items], sim_item), dim=0)
    return torch.mean(cos_loss_u) + torch.mean(cos_loss_i)



def proto_cl_loss(user_emb, user_indices, centroids, app2cluster, temperature):
    """
    Prototype contrastive loss: pull app embedding toward its text-cluster centroid,
    push away from other centroids.

    Args:
        user_emb: [n_users, dim] all user fused embeddings
        user_indices: [batch] user IDs in current batch
        centroids: [K, dim] cluster centroids (mean of fused embeddings per cluster)
        app2cluster: [n_users] LongTensor, cluster assignment for each app
        temperature: scalar
    """
    batch_emb = F.normalize(user_emb[user_indices], dim=1)        # [B, dim]
    norm_centroids = F.normalize(centroids, dim=1)                 # [K, dim]
    logits = torch.matmul(batch_emb, norm_centroids.T) / temperature  # [B, K]
    labels = app2cluster[user_indices]                             # [B]
    return F.cross_entropy(logits, labels)
