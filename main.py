import os, pickle
import numpy
import argparse
import csv
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import Dataset, DataLoader
from model import MMQF

'''Code for multi-class DDI prediction'''
# 创建了一个 ArgumentParser 对象，用于处理命令行参数。
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='which GPU card to use, 0/1/2/3 …')  # 服务器第几张卡
parser.add_argument('--hidden_dim', type=int, default=512, help='num of hidden features')  ##隐藏层的维度，即神经网络中隐藏层的节点数。
parser.add_argument('--out_dim', type=int, default=256, help='num of output features')  # 输出层的维度，即模型输出的特征数。
parser.add_argument('--dropout', type=float, default=0.5,help='dropout rate')  # Dropout 率，用于防止过拟合的一种正则化技术，通过随机丢弃一部分神经元来实现。
parser.add_argument('--rels', type=int, default=65,help='num of relations')  
parser.add_argument('--n_epochs', type=int, default=400, help='num of epochs')  
parser.add_argument('--batch_size', type=int, default=2048, help='batch size')  # 批量大小，即每次训练时使用的样本数量
parser.add_argument('--model', default='MMQF', type=str)  # 模型的类型，用于选择不同的模型架构
parser.add_argument('--filename', type=str, default='result.csv', help='folder to save results.csv')
parser.add_argument('--graph_alpha', type=float, default=0.6)#,choices=[0.6,0.7]
parser.add_argument('--min_sim', type=float, default=0.6)#,choices=[0.6,0.7]
parser.add_argument('--temperature', type=float, default=0.2, help='对比学习温度参数')
parser.add_argument('--cl_alpha', type=float, default=0.7)
parser.add_argument('--w_cl', type=float, default=0.1)#choices=[0.1,0.2]
parser.add_argument('--lr', type=float, default=3e-3)#,choices=[3e-3,2e-3]
parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
parser.add_argument('--fold', type=int, default=0, help='Which fold to use (0-4)')
parser.add_argument('--dataset', type=str, default='Deng', choices=['Deng','Ryu','Lin'], help='Dataset name')
args = parser.parse_args()  # 解析命令行输入的参数，并将它们存储在 args 对象中，以便在脚本的其他部分使用。


# 把原来的 device 选择逻辑改成
device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
print(f'[INFO] using device: {device}')
print(f"数据保存文件为：{args.filename},seed为{args.seed}fold为{args.fold}")

# ============================
# 3. 加载DDI数据
# ============================

print(f"读取数据: {args.dataset}/{args.seed}/{args.fold}/")
train_df = pd.read_csv(f'{args.dataset}/{args.seed}/{args.fold}/train.csv')
test_df = pd.read_csv(f'{args.dataset}/{args.seed}/{args.fold}/test.csv')

train_edges = train_df[['Drug1', 'Drug2']].values
train_labels = train_df['Label'].values
test_edges = test_df[['Drug1', 'Drug2']].values
test_labels = test_df['Label'].values

print(f"训练集: {len(train_edges)} 测试集: {len(test_edges)}")

# ============================
# 4. 加载相似度矩阵
# ============================

print("\n加载相似度矩阵...")
sim_mat_chem = np.loadtxt(f'{args.dataset}/molformer_sim_matrix.csv', delimiter=',')
jaccard_sim_matrix = np.loadtxt(f'{args.dataset}/chem_Jacarrd_sim.csv', delimiter=',')
brics_sim_matrix = np.load(f'{args.dataset}/brics_weighted_similarity_matrix.npy')

print(f"MolFormer 相似度矩阵: {sim_mat_chem.shape}")
print(f"Jaccard 相似度矩阵: {jaccard_sim_matrix.shape}")
print(f"BRICS 相似度矩阵: {brics_sim_matrix.shape}")

# ============================
# 5. 加载SMILES特征
# ============================

smiles_data = torch.load(f'{args.dataset}/Molformer_smiles_features.pt')
smiles_feat_mat = smiles_data.float().to(device)
args.smiles_input_dim = smiles_feat_mat.shape[1]
print(f"SMILES 特征维度: {args.smiles_input_dim}")





def generate_edges_weighted_threshold(
    sem_similarity_matrix,
    brics_similarity_matrix,
    real_edges,
    graph_alpha=args.graph_alpha,
    min_sim=args.min_sim
):
    """
    基于阈值的加权融合选边：
    score = graph_alpha * MolFormer + (1 - graph_alpha) * BRICS
    保留 score >= min_sim 的边，不使用 topk 限制。
    """
    if torch.is_tensor(sem_similarity_matrix):
        sem_similarity_matrix = sem_similarity_matrix.cpu().numpy()
    if torch.is_tensor(brics_similarity_matrix):
        brics_similarity_matrix = brics_similarity_matrix.cpu().numpy()

    num_nodes = sem_similarity_matrix.shape[0]

    real_set = {(int(u), int(v)) for u, v in real_edges}
    real_set.update({(int(v), int(u)) for u, v in real_edges})

    combined_sim = graph_alpha * sem_similarity_matrix + (1 - graph_alpha) * brics_similarity_matrix

    edges_set = set()
    stats = {
        "filtered_by_self_loop": 0,
        "filtered_by_ddi": 0,
        "filtered_by_threshold": 0,
        "final_edges": 0
    }

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                stats["filtered_by_self_loop"] += 1
                continue

            if (i, j) in real_set:
                stats["filtered_by_ddi"] += 1
                continue

            if combined_sim[i, j] < min_sim:
                stats["filtered_by_threshold"] += 1
                continue

            edges_set.add((i, j))

    stats["final_edges"] = len(edges_set)

    print(f"\n阈值加权融合选边:")
    print(f"  - graph_alpha: {graph_alpha}")
    print(f"  - min_sim: {min_sim}")
    print(f"  - 过滤自环: {stats['filtered_by_self_loop']}")
    print(f"  - 过滤已有DDI边: {stats['filtered_by_ddi']}")
    print(f"  - 阈值过滤: {stats['filtered_by_threshold']}")
    print(f"  - 最终保留边: {stats['final_edges']}")

    if len(edges_set) == 0:
        return torch.empty((0, 2), dtype=torch.long), stats

    return torch.tensor(list(edges_set), dtype=torch.long), stats



def build_unified_graph(train_edges,train_labels,sim_edges,num_ddi_relations):
    """构建统一异构图"""
    # 原始无向DDI边
    ddi_edges = train_edges
    # 为PyG消息传播补充反向边
    ddi_edges_rev = train_edges[:, [1, 0]]
    ddi_types = train_labels
    ddi_types_rev = train_labels
    sim_edges_np = (sim_edges.cpu().numpy()if torch.is_tensor(sim_edges)else np.asarray(sim_edges))

    # 再次过滤与训练DDI边重叠的相似边
    if len(sim_edges_np) > 0:
        ddi_set = set()
        for u, v in train_edges:
            ddi_set.add((int(u), int(v)))
            ddi_set.add((int(v), int(u)))
        filtered = [
            e for e in sim_edges_np
            if (int(e[0]), int(e[1])) not in ddi_set]

        sim_edges_np = (np.asarray(filtered, dtype=np.int64)
            if filtered
            else np.empty((0, 2), dtype=np.int64))

    if len(sim_edges_np) > 0:
        # 相似边生成阶段已经包含(A,B)和(B,A)
        # 只进行有向边去重，不再构造sim_edges_rev
        sim_edges_np = np.unique(sim_edges_np.astype(np.int64),axis=0)

        sim_types = np.full(len(sim_edges_np),num_ddi_relations,dtype=np.int64)

        unified_edges = np.vstack([ddi_edges,ddi_edges_rev,sim_edges_np])

        unified_types = np.concatenate([ddi_types,ddi_types_rev,sim_types])

    else:
        unified_edges = np.vstack([ddi_edges,ddi_edges_rev ])
        unified_types = np.concatenate([ddi_types,ddi_types_rev])

    edge_index = torch.tensor(unified_edges,dtype=torch.long).t().contiguous().to(device)

    edge_type = torch.tensor(unified_types,dtype=torch.long).to(device)

    print(
        f"统一图构建完成: "
        f"DDI边={len(ddi_edges) * 2}, "
        f"相似边={len(sim_edges_np)}")
    return edge_index, edge_type

# ============================
# 构建原始图（仅DDI边，无语义边）
# ============================

def build_original_graph(train_edges, train_labels, num_ddi_relations):
    """
    构建原始DDI图（无语义增强边）
    - 只有DDI边: type = 0~64
    """
    # DDI 边 (正向 + 反向)
    ddi_edges = train_edges
    ddi_edges_rev = train_edges[:, [1, 0]]
    ddi_types = train_labels
    ddi_types_rev = train_labels

    unified_edges = np.vstack([ddi_edges, ddi_edges_rev])
    unified_types = np.concatenate([ddi_types, ddi_types_rev])

    edge_index = torch.tensor(unified_edges, dtype=torch.long).t().contiguous().to(device)
    edge_type = torch.tensor(unified_types, dtype=torch.long).to(device)

    print(f"原始图构建完成: DDI边={len(ddi_edges) * 2}, 无语义边")
    return edge_index, edge_type
# 构建原始图
original_edge_index, original_edge_type = build_original_graph(train_edges, train_labels, num_ddi_relations=args.rels)
all_real_edges = np.vstack([train_edges, test_edges])

new_edges_sim, _ = generate_edges_weighted_threshold(
        sem_similarity_matrix=sim_mat_chem,
        brics_similarity_matrix=brics_sim_matrix,
        real_edges=all_real_edges)
unified_edge_index, unified_edge_type = build_unified_graph(
    train_edges, train_labels, new_edges_sim, num_ddi_relations=args.rels)


# ============================
# 7. 加载BRICS片段数据（统一处理）
# ============================

def load_brics_data(brics_pkl_path=f'{args.dataset}/brics_fragments_data.pkl',
                    gin_graph_pkl_path=f'{args.dataset}/fragment_graph_data.pkl',
                    gin_indices_pkl_path=f'{args.dataset}/drug_fragment_indices.pkl'):
    """
    一次性加载所有BRICS相关数据，返回设备上的Tensor
    """
    # 加载原始片段数据
    with open(brics_pkl_path, 'rb') as f:
        brics_data = pickle.load(f)

    drug_fragments = brics_data['drug_fragments']
    fragment_idf = brics_data['fragment_idf']
    drug_list = list(drug_fragments.keys())
    num_drugs = len(drug_list)

    # 构建词汇表
    # 为每个药物准备片段索引和IDF
    # 加载GIN预处理数据
    with open(gin_graph_pkl_path, 'rb') as f:
        gin_graph_data = pickle.load(f)
    with open(gin_indices_pkl_path, 'rb') as f:
        gin_indices_data = pickle.load(f)

    # Use the exact vocabulary and drug indices that were used to build frag_graphs.
    frag_to_idx = gin_graph_data['frag_to_idx']
    frag_smiles = gin_graph_data['frag_smiles']
    saved_drug_indices = gin_indices_data['drug_indices']

    canonical_fragment_idf = {}
    for frag, idf in fragment_idf.items():
        mol = Chem.MolFromSmiles(frag)
        canonical_frag = Chem.MolToSmiles(mol) if mol is not None else frag
        canonical_fragment_idf[canonical_frag] = idf

    drug_indices = []
    drug_idfs = []
    for drug_id in drug_list:
        indices = list(saved_drug_indices.get(drug_id, []))
        if any(idx < 0 or idx >= len(frag_smiles) for idx in indices):
            raise ValueError(f'{drug_id} contains an invalid BRICS fragment index')
        idfs = [canonical_fragment_idf.get(frag_smiles[idx], 1.0) for idx in indices]
        drug_indices.append(indices)
        drug_idfs.append(idfs)

    frag_graphs = gin_graph_data['frag_graphs']
    max_fragments = gin_indices_data['max_fragments']
    node_feat_dim = gin_graph_data['node_feat_dim']
    edge_feat_dim = gin_graph_data['edge_feat_dim']

    # 构建矩阵（先CPU，后统一移到GPU）
    num_drugs = len(drug_list)
    fragment_indices_matrix = torch.zeros(num_drugs, max_fragments, dtype=torch.long)
    fragment_idfs_matrix = torch.zeros(num_drugs, max_fragments, dtype=torch.float32)
    fragment_mask_matrix = torch.zeros(num_drugs, max_fragments, dtype=torch.bool)

    for i, (indices, idfs) in enumerate(zip(drug_indices, drug_idfs)):
        n = min(len(indices), max_fragments)
        if n > 0:
            fragment_indices_matrix[i, :n] = torch.tensor(indices[:n], dtype=torch.long)
            fragment_idfs_matrix[i, :n] = torch.tensor(idfs[:n], dtype=torch.float32)
            fragment_mask_matrix[i, :n] = True

    # 构建GIN图数据（先CPU，后统一移到GPU）
    frag_graphs_gpu = []
    for graph in frag_graphs:
        frag_graphs_gpu.append({
            'x': graph['x'].clone(),
            'edge_index': graph['edge_index'].clone(),
            'edge_attr': graph['edge_attr'].clone()
        })

    print(f"\nBRICS数据加载完成:")
    print(f"  - 药物数: {num_drugs}")
    print(f"  - 唯一片段数: {len(frag_to_idx)}")
    print(f"  - 最大片段数: {max_fragments}")
    print(f"  - 节点特征维度: {node_feat_dim}")
    print(f"  - 边特征维度: {edge_feat_dim}")
    print(f"  - 有效片段占比: {fragment_mask_matrix.sum().item() / (num_drugs * max_fragments):.2%}")

    return {
        'drug_list': drug_list,
        'frag_to_idx': frag_to_idx,
        'fragment_indices': fragment_indices_matrix,
        'fragment_idfs': fragment_idfs_matrix,
        'fragment_mask': fragment_mask_matrix,
        'frag_graphs': frag_graphs_gpu,
        'node_feat_dim': node_feat_dim,
        'edge_feat_dim': edge_feat_dim,
        'max_fragments': max_fragments,
        'num_drugs': num_drugs
    }


# 加载所有BRICS数据
brics_data = load_brics_data()

# 移动到GPU
fragment_indices_matrix = brics_data['fragment_indices'].to(device)
fragment_idfs_matrix = brics_data['fragment_idfs'].to(device)
fragment_mask_matrix = brics_data['fragment_mask'].to(device)

# GIN图数据移到GPU
frag_graphs_gpu = []
for graph in brics_data['frag_graphs']:
    frag_graphs_gpu.append({
        'x': graph['x'].to(device),
        'edge_index': graph['edge_index'].to(device),
        'edge_attr': graph['edge_attr'].to(device)
    })

print(f"\n✅ 所有数据已移动到 {device}")
print(f"  - fragment_indices: {fragment_indices_matrix.shape}")
print(f"  - fragment_idfs: {fragment_idfs_matrix.shape}")
print(f"  - fragment_mask: {fragment_mask_matrix.shape}")
print(f"  - GIN图数量: {len(frag_graphs_gpu)}")


class DDIDataset(Dataset):
    '''Customized dataset processing class'''

    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


# 创建 Dataset 和 Loader
train_dataset = DDIDataset(train_edges, train_labels)
test_dataset = DDIDataset(test_edges, test_labels)

train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=args.batch_size)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size)



if args.model == 'MMQF':
    model = MMQF(
        args=args,
        frag_graphs=frag_graphs_gpu,
        node_feat_dim=brics_data['node_feat_dim'],
        edge_feat_dim=brics_data['edge_feat_dim'],
        frag_hidden_dim=128,
        frag_output_dim=256,
        cross_hidden_dim=256,
        cross_heads=4
    )
else:
    raise ValueError(f"Unknown model: {args.model}")


print(f"使用的模型为{args.model}")  # 将模型移动到之前设置的设备（CPU或GPU）上。
model.to(device)


def contrastive_loss(v1, v2):
    """
    计算两个视图嵌入之间的对比损失 (NT-Xent Loss 的简化版)
    """
    v1 = F.normalize(v1, dim=1)
    v2 = F.normalize(v2, dim=1)
    # 计算余弦相似度矩阵 [N, N]
    sim_matrix = torch.matmul(v1, v2.T) / args.temperature
    # 对角线为正样本对
    labels = torch.arange(v1.size(0)).to(v1.device)
    loss = (F.cross_entropy(sim_matrix, labels) +
            F.cross_entropy(sim_matrix.T, labels)) / 2
    return loss


def train_():
    criterion = nn.CrossEntropyLoss().to(device)  # 交叉熵损失函数，用于多分类问题
    re_criterion = nn.MSELoss().to(device)  # 均方误差损失函数，评估重构输出（如自编码器）与原始输入的误差
    trainer_E = torch.optim.Adam(model.parameters(), lr=args.lr)  # 使用 Adam 优化器来更新模型参数，学习率为 1e-3
    n_iterations = len(train_loader)  # 每个 epoch 中的迭代次数，等于训练数据加载器的长度。
    num_epochs = args.n_epochs  # 总的训练轮数

    # Training phase
    for epoch in range(num_epochs):
        model.train()
        true_labels, pred_labels = [], []
        running_loss = 0.0  # 累计损失
        running_correct = 0.0  # 累计正确分类的数量
        total_samples = 0  # 总样本数
        for i, (edges, labels) in enumerate(train_loader):
            edges, labels = edges.to(device), labels.to(device)  # 迁移到GPU
            y_pred, (z_sem, z_graph, z_frag,pair_embedding, attn_weights) = model(unified_edge_index, unified_edge_type,
                                        smiles_feat_mat,fragment_indices_matrix, fragment_idfs_matrix, fragment_mask_matrix,edges)
            c_loss = args.cl_alpha * contrastive_loss(z_sem, z_graph) + (1 - args.cl_alpha) * contrastive_loss(z_frag, z_graph)

            cls_loss = criterion(y_pred, labels)
            total_loss = cls_loss + args.w_cl * c_loss * 2

            trainer_E.zero_grad()  # 清空历史梯度
            total_loss.backward()  # 通过计算图反向传播梯度（自动微分）
            trainer_E.step()  # 根据梯度方向更新参数（Adam自适应调整学习率）

            running_correct += torch.sum(
                (torch.argmax(y_pred, dim=1).type(torch.FloatTensor) == labels.cpu()).detach()).float()
            # 将当前批次的预测标签添加到 pred_labels 列表中。
            pred_labels.append(list(y_pred.cpu().detach().numpy().reshape(-1)))
            labels = labels.cpu().numpy()
            # 将当前批次的真实标签添加到 true_labels 列表中。
            true_labels.append(list(labels))
            # labels.shape[0]：获取当前批次的样本数量，total_samples += ...：累计总样本数量
            total_samples += labels.shape[0]
            # 获取当前批次的损失值，并累计总损失
            running_loss += total_loss.item()
        print(f"epoch {epoch + 1}/{num_epochs};trainging loss: {running_loss / n_iterations:.4f} ;training set acc: {running_correct / total_samples:.4f}")

    test_(test_loader , args.n_epochs)


# y_true = np.array([[0, 1], [1, 0], [0, 1]])  # 2个类别的真实标签
# y_score = np.array([[0.2, 0.8], [0.6, 0.4all.csv], [0.3, 0.7]]) # 预测概率
def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)

    # 计算多类分类问题的平均分数。
    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()

        # 确保输入为二维数组（n_samples × n_classes）
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))

        # 逐类别计算AUPR
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()  # 取第c类的真实标签
            y_score_c = y_score.take([c], axis=1).ravel()  # 取第c类的预测概率
            score[c] = binary_metric(y_true_c, y_score_c)  # 计算该类AUPR
        return np.average(score)

    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)


def test_(loader,  best_epoch):
    event_num = args.rels  # 类别的数量，这里假设有65个不同的事件或类别
    n_test_samples = 0  # 测试样本的总数
    n_correct = 0  # 正确预测的样本数
    total_labels = []  # 存储所有测试样本的真实标签
    total_pred = []  # 存储所有测试样本的预测概率

    # Testing phase
    with torch.no_grad():  # 确保在评估过程中不计算梯度，节省计算资源。
        model.eval()  # 将模型设置为评估模式
        for edges, labels in loader:
            edges, labels = edges.to(device), labels.to(device)
            y_pred, _ = model(unified_edge_index, unified_edge_type,smiles_feat_mat,
                fragment_indices_matrix, fragment_idfs_matrix, fragment_mask_matrix,edges)

            y_hat = F.softmax(y_pred, dim=1)
            total_pred.append(y_hat.cpu().numpy())
            total_labels.append(labels.cpu().numpy())
            n_test_samples += edges.shape[0]
            n_correct += torch.sum((torch.argmax(y_pred, dim=1).type(torch.FloatTensor) == labels.cpu()).detach()).float()

        total_pred = np.vstack(total_pred)  # 合并所有批次的预测概率
        total_labels = np.concatenate(total_labels)  # 合并真实标签
        pred_type = np.argmax(total_pred, axis=1)  # 获取预测类别 ([[0.2, 0.8], [0.6, 0.4all.csv], [0.3, 0.7]]) # 预测概率
        y_one_hot = label_binarize(y=total_labels,
                                   classes=np.arange(event_num))  # ([[0, 1], [1, 0], [0, 1]])  # 2个类别的真实标签

        acc = n_correct / n_test_samples  # 计算总体准确率。
        # y_true = np.array([[0, 1], [1, 0], [0, 1]])  # 2个类别的真实标签
        # y_score = np.array([[0.2, 0.8], [0.6, 0.4all.csv], [0.3, 0.7]]) # 预测概率
        aupr = roc_aupr_score(y_one_hot, total_pred, average='micro')
        auroc = roc_auc_score(y_one_hot, total_pred, average='micro')
        f1 = f1_score(total_labels, pred_type, average='macro')
        precision = precision_score(total_labels, pred_type, average='macro', zero_division=0)
        recall = recall_score(total_labels, pred_type, average='macro', zero_division=0)

        # print(f"{acc}\t{aupr}\t{auroc}{f1}\t{precision}\t{recall}")
        print(f"-------test set result---------")
        print(f"ACC: {acc}")
        print(f"AUPR: {aupr}")
        print(f"AUROC: {auroc}")
        print(f"F1: {f1}")
        print(f"Precison: {precision}")
        print(f"Recall: {recall}")
        if best_epoch != 0:
            save_results(best_epoch, acc, aupr, auroc, f1, precision, recall)

        return f1


def save_results(best_epoch, acc, aupr, auroc, f1, precision, recall):
    try:
        with open(args.filename, 'r') as f:
            pass
    except FileNotFoundError:
        with open(args.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["model", 'seed', 'fold', 'batch_size', 'n_epochs', 'graph_alpha', 'min_sim',
                            'w_cl', 'cl_alpha', 'temperature','dropout', 'lr',
                             'ACC', 'AUPR', 'AUROC', 'F1', 'Precision', 'Recall'])

    with open(args.filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([args.model,args.seed, args.fold, args.batch_size, best_epoch,
            args.graph_alpha, args.min_sim,args.w_cl, args.cl_alpha, args.temperature,
            args.dropout, args.lr, f"{acc:.6f}", f"{aupr:.6f}", f"{auroc:.6f}",
            f"{f1:.6f}", f"{precision:.6f}", f"{recall:.6f}"
        ])


train_()


