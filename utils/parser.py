import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate TEARec.")
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')
    parser.add_argument('--Lambda', type=int, default=8,
                        help='choose the degree of the neighbors similiar to themselves')
    parser.add_argument('--alpha1', type=float, default=0.53,
                        help='choose the degree of the neighbors similiar to themselves')
    parser.add_argument('--alpha2', type=float, default=0.48,
                        help='choose the degree of the neighbors similiar to themselves')
    parser.add_argument('--data_path', nargs='?', default='datasets/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')
    parser.add_argument('--save_recom', type=int, default=0,
                        help='Whether save the recommendation results.')
    parser.add_argument('--model_name', nargs='?', default='TEARec',
                        help='Model name used in log filenames.')
    parser.add_argument('--dataset', nargs='?', default='MLO-3',
                        help='Choose a dataset from given folder')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=800,
                        help='Number of epoch.')
    parser.add_argument('--embed_size', type=int, default=128,
                        help='Embedding size.')
    parser.add_argument('--layer_size', type=int, default=3,
                        help='Number of LightGCN propagation layers.')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size.')
    parser.add_argument('--regs', nargs='?', default='[2e-3]',
                        help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--cl_reg', type=float, default=0.004,
                        help='Weight of contrastive learning loss. Set to 0 to disable.')
    parser.add_argument('--ssl_reg', type=float, default=0.0001,
                        help='Weight of implicit supervision loss (coefficient-weighted similarity). Set to 0 to disable.')
    parser.add_argument('--cl_temp', type=float, default=0.15,
                        help='Temperature parameter for contrastive learning.')
    parser.add_argument('--cl_degree_pct', type=float, default=0.2,
                        help='Only apply CL to top-X%% degree nodes (app/tpl separately). Default: 0.2 (top 20%%).')
    parser.add_argument('--loss_type', nargs='?', default='ssm',
                        help='Main loss type: bpr / ssm. (ssm is the sampled softmax mode)')
    parser.add_argument('--khop', type=int, default=1,
                        help='k-hop on bipartite graph to exclude in ssm loss (odd: 1,3,5).')
    parser.add_argument('--softmax_temp', type=float, default=0.5,
                        help='Temperature for ssm loss.')
    parser.add_argument('--ssm_neg_count', type=int, default=64,
                        help='Number of sampled negatives per user for ssm loss.')
    parser.add_argument('--enable_tail_app_aug', type=int, default=1,
                        help='Enable rule-based tail-tpl to app augmentation on the interaction graph. 0: disable, 1: enable.')
    parser.add_argument('--aug_lambda', type=float, default=2.0,
                        help='Edge weight for augmented tail-tpl to app edges.')
    parser.add_argument('--alpha_init', type=float, default=0.5,
                        help='Initial fusion weight for the homo branch. Range: (0,1).')
    parser.add_argument('--homo_only', type=int, default=0,
                        help='Ablation mode: use only similarity graphs for recommendation. Enables rec+ssl only.')
    parser.add_argument('--hetero_only', type=int, default=0,
                        help='Ablation mode: use only interaction graph for recommendation. Disables similarity-based auxiliary losses.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1,0.1,0.1,0.1,0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. ')
    parser.add_argument('--drop_edge',type=float,default=0.95,
                        help="perserve the percent of edges")
    parser.add_argument('--Ks', nargs='?', default='[5,10]',
                        help='Evaluation cutoffs, for example [5,10].')
    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument('--eval_mode', nargs='?', default='gpu',
                        help='Evaluation implementation: gpu / cpu. gpu uses the vectorized evaluator, cpu uses the CPU/heap-based evaluator.')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')

    parser.add_argument('--seed', type=int, default=2026,
                        help='Random seed for reproducibility. Default: 2026')

    parser.add_argument('--tensorboard', type=int, default=0,
                        help='Enable TensorBoard logging. 0: Disable, 1: Enable (Default: 1)')

    # SSL loss exponent parameters
    parser.add_argument('--ssl_coeff_power', type=float, default=3.0,
                        help='Exponent for SSL coefficient (applied after ReLU). Default: 3.0')
    parser.add_argument('--ssl_sim_power', type=float, default=4.0,
                        help='Exponent for SSL similarity (applied after ReLU). Default: 4.0')

    # Prototype contrastive learning (text-based KMeans)
    parser.add_argument('--proto_reg', type=float, default=0.001,
                        help='Weight of prototype contrastive learning loss. 0=disabled.')
    parser.add_argument('--n_clusters', type=int, default=256,
                        help='Number of KMeans clusters for text-based prototype CL.')
    parser.add_argument('--proto_temp', type=float, default=0.1,
                        help='Temperature for prototype contrastive loss.')
    parser.add_argument('--cluster_mode', nargs='?', default='pca',
                        help='Prototype clustering input: raw / pca.')
    parser.add_argument('--cluster_pca_dim', type=int, default=128,
                        help='PCA dimension before KMeans when cluster_mode=pca.')
    parser.add_argument('--bert_model', nargs='?', default='BAAI/bge-large-en-v1.5',
                        help='BERT model name or path for text encoding.')
    parser.add_argument('--bert_max_length', type=int, default=256,
                        help='Max token length for BERT encoding.')
    parser.add_argument('--bert_batch_size', type=int, default=32,
                        help='Batch size for BERT text encoding.')

    return parser.parse_args()
