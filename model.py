import numpy as np
from torch_geometric.nn import RGCNConv, SAGEConv, GCNConv, GraphConv,GINConv, global_mean_pool

import torch.nn.functional as F #，提供了各种神经网络层和函数的实现，如激活函数、池化函数等
import torch
import torch.nn as nn #提供了构建神经网络所需的各种层和模块，如全连接层、卷积层、循环层等。
from torch.nn import MultiheadAttention
import pickle
from collections import defaultdict
from torch_geometric.data import Data, Batch
class AutoEncoder(nn.Module):
    # input_dim：输入数据的维度。hidden_dim：隐藏层的维度。latent_dim：潜在空间的维度（编码后的表示维度）
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(AutoEncoder, self).__init__()

        # 编码器
        # 将输入维度映射到两倍的隐藏维度。
        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ELU(),                                    #  指数线性单元（Exponential Linear Unit）激活函数
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(0.2),
        )

        # 将两倍的隐藏维度映射到隐藏维度。
        self.encoder2 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
        )

        # 将隐藏维度映射到潜在维度。
        self.encoder3 = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.ELU(),
            nn.BatchNorm1d(latent_dim),
        )

        # 解码器
        #解码器将潜在维度映射回原始输入维度
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ELU(),
            nn.Linear(hidden_dim * 2, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z1 = self.encoder1(x)
        z2 = self.encoder2(z1)
        z3 = self.encoder3(z2)
        reconstructed = self.decoder(z3)  #解码器的输出，即重构的输入数据
        return torch.hstack([z1, z2, z3]), reconstructed #返回编码器各层的输出和重构的输入数据。



class R_GCN(torch.nn.Module):
    def __init__(self, num_feat, num_outputs, num_hidden, dropout, rels):
        super(R_GCN, self).__init__()
        # Define two conv layers

        self.conv1 = RGCNConv(num_feat, num_hidden * 2, num_relations=rels)
        self.conv2 = RGCNConv(num_hidden * 2, num_hidden, num_relations=rels)
        self.conv3 = RGCNConv(num_hidden, num_outputs, num_relations=rels)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type):
        h1 = self.conv1(x, edge_index, edge_type)
        h1 = F.elu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = self.conv2(h1, edge_index, edge_type)
        h2 = F.elu(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        h3 = self.conv3(h2, edge_index, edge_type)
        h3 = F.elu(h3)
        return torch.hstack([h1, h2, h3])


class DNN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DNN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 512),
            nn.ELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_outputs)
        )

    def forward(self, x):
        output = self.layers(x)
        return output




# ============================================================
# 优化3: Dense Connection（稠密连接）
# ============================================================

class SemanticEncoderV3_Dense(nn.Module):
    """
    多层级语义编码器 + Dense连接（DenseNet风格）

    每层都接收前面所有层的信息，实现最大程度的信息复用

    输入: x [N, 768] MolFormer特征
    输出:
        h_sem [N, 1792] 融合后的语义表示
        (z1, z2, z3) 各尺度输出，用于后续分析
    """

    def __init__(self, input_dim=768, hidden_dim=512, dropout=0.2):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ELU(),
            nn.Dropout(dropout)
        )

        # enc2: 输入 = z1 + x (保留原始信息)
        self.enc2 = nn.Sequential(
            nn.Linear(1024 + input_dim, 512),
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Dropout(dropout)
        )

        # enc3: 输入 = z1 + z2 + x (保留所有前层信息)
        self.enc3 = nn.Sequential(
            nn.Linear(1024 + 512 + input_dim, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        z1 = self.enc1(x)  # [N, 1024]

        # enc2 输入: z1 + x (concat)
        z2_input = torch.cat([z1, x], dim=-1)  # [N, 1024+768=1792]
        z2 = self.enc2(z2_input)  # [N, 512]

        # enc3 输入: z1 + z2 + x (concat所有前层)
        z3_input = torch.cat([z1, z2, x], dim=-1)  # [N, 1024+512+768=2304]
        z3 = self.enc3(z3_input)  # [N, 256]

        h_sem = torch.cat([z1, z2, z3], dim=-1)  # [N, 1024+512+256=1792]

        return h_sem, (z1, z2, z3)





# ============================================================
# 基础组件：GIN Fragment Encoder（含各种优化选项）
# ============================================================

class BaseGINFragmentEncoder(nn.Module):
    """
    基础GIN编码器（当前正在使用的版本）
    作为所有优化实验的基线
    """
    def __init__(self, frag_graphs, node_feat_dim=6, edge_feat_dim=5,
                 hidden_dim=128, output_dim=256, dropout=0.3):
        super().__init__()

        self.frag_graphs = frag_graphs
        self.num_frags = len(frag_graphs)
        self.output_dim = output_dim

        # GIN layers
        self.gin1 = GINConv(nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))

        self.gin2 = GINConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))

        self.proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("frag_embeddings", torch.zeros(self.num_frags, output_dim))
        self._build_embeddings()

    @torch.no_grad()
    def _build_embeddings(self):
        for i in range(self.num_frags):
            g = self.frag_graphs[i]
            x = g["x"]
            edge_index = g["edge_index"]
            if x.device != self.frag_embeddings.device:
                x = x.to(self.frag_embeddings.device)
                edge_index = edge_index.to(self.frag_embeddings.device)

            h = self.gin1(x, edge_index)
            h = F.relu(h)
            h = self.dropout(h)

            h = self.gin2(h, edge_index)
            h = F.relu(h)

            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
            h = global_mean_pool(h, batch)
            h = self.proj(h)
            self.frag_embeddings[i] = h.squeeze()
        print(f"✔ Base GIN Fragment table built: {self.frag_embeddings.shape}")

    def forward(self, fragment_indices, fragment_mask):
        frag_tokens = self.frag_embeddings[fragment_indices]
        frag_tokens = frag_tokens * fragment_mask.unsqueeze(-1)

        denom = fragment_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        h_frag_pooled = frag_tokens.sum(dim=1) / denom

        return frag_tokens, h_frag_pooled



# ============================================================
# 优化5: Jumping Knowledge（跳接）
# ============================================================

class GINFragmentEncoder_JK(BaseGINFragmentEncoder):
    """
    Jumping Knowledge: 拼接所有GIN层的输出
    """

    def __init__(self, frag_graphs, node_feat_dim=6, edge_feat_dim=5,
                 hidden_dim=128, output_dim=256, dropout=0.3):
        super().__init__(frag_graphs, node_feat_dim, edge_feat_dim, hidden_dim, output_dim, dropout)

        # 重写GIN结构
        self.gin1 = GINConv(nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))
        self.gin2 = GINConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))
        # JK投影层：hidden_dim*2 → output_dim
        self.proj = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("frag_embeddings", torch.zeros(self.num_frags, output_dim))
        self._build_embeddings_jk()

    @torch.no_grad()
    def _build_embeddings_jk(self):
        """JK: 拼接所有层输出"""
        for i in range(self.num_frags):
            g = self.frag_graphs[i]
            x = g["x"]
            edge_index = g["edge_index"]
            if x.device != self.frag_embeddings.device:
                x = x.to(self.frag_embeddings.device)
                edge_index = edge_index.to(self.frag_embeddings.device)

            h1 = self.gin1(x, edge_index)
            h1 = F.relu(h1)
            h1 = self.dropout(h1)

            h2 = self.gin2(h1, edge_index)
            h2 = F.relu(h2)

            batch = torch.zeros(h1.size(0), dtype=torch.long, device=h1.device)
            h1_pool = global_mean_pool(h1, batch)
            h2_pool = global_mean_pool(h2, batch)

            # 拼接 + 投影
            h = self.proj(torch.cat([h1_pool, h2_pool], dim=-1))
            self.frag_embeddings[i] = h.squeeze()
        print(f"✔ JK GIN Fragment table built: {self.frag_embeddings.shape}")

    def forward(self, fragment_indices, fragment_mask):
        frag_tokens = self.frag_embeddings[fragment_indices]
        frag_tokens = frag_tokens * fragment_mask.unsqueeze(-1)

        denom = fragment_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        h_frag_pooled = frag_tokens.sum(dim=1) / denom

        return frag_tokens, h_frag_pooled








# ============================================================
# 优化2: Multi-Query Cross-Attention (K=4)
# ============================================================

class MultiQueryCrossAttentionFusion(nn.Module):
    """
    Multi-Query Cross-Attention: 多个可学习Query从不同角度关注片段
    """

    def __init__(self, sem_dim=1792, frag_dim=256, hidden_dim=256,
                 num_heads=4, num_queries=4, dropout=0.1):
        super().__init__()

        self.num_queries = num_queries
        self.sem_proj = nn.Linear(sem_dim, hidden_dim * num_queries)
        self.frag_proj = nn.Linear(frag_dim, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * num_queries, sem_dim),
            nn.LayerNorm(sem_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_sem, frag_tokens, frag_mask):
        B, F, _ = frag_tokens.shape
        key_padding_mask = ~frag_mask

        # 生成多个Query [B, K, D]
        q = self.sem_proj(h_sem).view(B, self.num_queries, -1)
        kv = self.frag_proj(frag_tokens)  # [B, F, D]

        attn_out, attn_weights = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask)

        # 展平多个Query的输出
        h_fused = h_sem + self.out_proj(attn_out.flatten(1))

        return h_fused, attn_weights





class EnhancedPairPredictor(nn.Module):
    """
    增强的药物对预测器
    输入: [h, t, |h-t|, h⊙t] ∈ R^(4*dim)
    """

    def __init__(self, input_dim, num_classes, dropout):
        super(EnhancedPairPredictor, self).__init__()

        self.predictor = nn.Sequential(
            nn.Linear(input_dim * 4, 512),
            nn.ELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, h, t):
        """
        Args:
            h: [batch, dim] 头节点表示
            t: [batch, dim] 尾节点表示
        Returns:
            logits: [batch, num_classes]
        """
        pair_feat = torch.cat([
            h,
            t,
            torch.abs(h - t),
            h * t
        ], dim=-1)
        return self.predictor(pair_feat)









# ============================================================
# 三模块组合：F1 + C + S2（JK + MultiQuery + Dense）
# ============================================================

class Model_F1_C_S2(nn.Module):
    """F1 (JK) + C (MultiQuery) + S2 (Dense)"""
    def __init__(self, args, frag_graphs, node_feat_dim=6, edge_feat_dim=5,
                 frag_hidden_dim=128, frag_output_dim=256,
                 cross_hidden_dim=256, cross_heads=4, num_queries=4):
        super().__init__()

        self.semantic_encoder = SemanticEncoderV3_Dense(
            input_dim=args.smiles_input_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout
        )

        self.fragment_encoder = GINFragmentEncoder_JK(
            frag_graphs=frag_graphs,
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=frag_hidden_dim,
            output_dim=frag_output_dim,
            dropout=args.dropout
        )

        self.cross_fusion = MultiQueryCrossAttentionFusion(
            sem_dim=1792,
            frag_dim=frag_output_dim,
            hidden_dim=cross_hidden_dim,
            num_heads=cross_heads,
            num_queries=num_queries,
            dropout=args.dropout
        )

        self.rgcn = R_GCN(
            num_feat=1792,
            num_hidden=args.hidden_dim,
            num_outputs=args.out_dim,
            dropout=args.dropout,
            rels=args.rels + 1
        )

        self.projection = nn.Sequential(
            nn.Linear(1792, 512), nn.ELU(), nn.Dropout(args.dropout),
            nn.Linear(512, args.out_dim), nn.LayerNorm(args.out_dim)
        )

        self.predictor = EnhancedPairPredictor(
            input_dim=args.out_dim, num_classes=args.rels, dropout=args.dropout
        )

        self.sem_cons_proj = nn.Sequential(nn.Linear(1792, 512), nn.ELU(), nn.Linear(512, 256))
        self.graph_cons_proj = nn.Sequential(nn.Linear(1792, 512), nn.ELU(), nn.Linear(512, 256))
        self.frag_cons_proj = nn.Sequential(nn.Linear(frag_output_dim, 512), nn.ELU(), nn.Linear(512, 256))

        self.temperature = args.temperature

    def forward(self, unified_edge_index, unified_edge_type, smiles_matrix,
                fragment_indices, fragment_idfs, fragment_mask, triples):

        h_sem, _ = self.semantic_encoder(smiles_matrix)
        frag_tokens, h_frag_pooled = self.fragment_encoder(fragment_indices, fragment_mask)

        # h_fused, _ = self.cross_fusion(h_sem, frag_tokens, fragment_mask)
        h_fused, attn_weights = self.cross_fusion(h_sem, frag_tokens, fragment_mask)

        h_graph = self.rgcn(h_fused, unified_edge_index, unified_edge_type)
        node_emb_proj = self.projection(h_graph)

        z_sem = self.sem_cons_proj(h_sem)
        z_graph = self.graph_cons_proj(h_graph)
        z_frag = self.frag_cons_proj(h_frag_pooled)

        triples = triples.long()
        # h = node_emb_proj[triples[:, 0]]
        # t = node_emb_proj[triples[:, 1]]
        # output = self.predictor(h, t)
        #
        # return output, (z_sem, z_graph, z_frag)
        h = node_emb_proj[triples[:, 0]]
        t = node_emb_proj[triples[:, 1]]

        pair_embedding = torch.cat([h, t, torch.abs(h - t), h * t], dim=-1)

        output = self.predictor(h, t)

        return output, (z_sem, z_graph, z_frag, pair_embedding, attn_weights)






class GINFragmentEncoder_JK1(nn.Module):
    """
    可端到端训练的JK-GIN片段编码器。

    输入：
        frag_graphs:
            所有BRICS片段图组成的列表。
            每个元素包含:
                x           原子节点特征
                edge_index  化学键连接关系
                edge_attr   化学键特征，当前版本暂不使用

        fragment_indices:
            shape = [num_drugs, max_fragments]
            每个药物包含的片段编号

        fragment_mask:
            shape = [num_drugs, max_fragments]
            标记有效片段和补零位置

    输出：
        frag_tokens:
            shape = [num_drugs, max_fragments, output_dim]
            每个药物对应的片段token

        h_frag_pooled:
            shape = [num_drugs, output_dim]
            每个药物的平均片段表示
    """

    def __init__(
        self,
        frag_graphs,
        node_feat_dim=6,
        edge_feat_dim=5,
        hidden_dim=128,
        output_dim=256,
        dropout=0.3
    ):
        super().__init__()

        self.num_frags = len(frag_graphs)
        self.output_dim = output_dim

        # ====================================================
        # 1. 将所有独立片段图合并成一个PyG Batch
        # ====================================================

        frag_data_list = []

        for graph in frag_graphs:
            data = Data(
                x=graph["x"].float(),
                edge_index=graph["edge_index"].long(),
                edge_attr=graph["edge_attr"].float()
            )
            frag_data_list.append(data)

        frag_batch = Batch.from_data_list(frag_data_list)

        # 将固定的片段图数据注册为buffer：
        # 它们会随model.to(device)一起移动，
        # 但不会被优化器更新。
        self.register_buffer(
            "frag_x",
            frag_batch.x
        )

        self.register_buffer(
            "frag_edge_index",
            frag_batch.edge_index
        )

        self.register_buffer(
            "frag_batch_index",
            frag_batch.batch
        )

        # ====================================================
        # 2. 第一层GIN
        # ====================================================

        self.gin1 = GINConv(
            nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        )

        # ====================================================
        # 3. 第二层GIN
        # ====================================================

        self.gin2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        )

        # ====================================================
        # 4. Jumping Knowledge投影
        # ====================================================
        # 第一层图表示和第二层图表示进行拼接：
        # hidden_dim + hidden_dim = hidden_dim * 2
        # 再投影为output_dim
        # ====================================================

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )

        self.dropout = nn.Dropout(dropout)

        print(
            f"Trainable JK-GIN initialized: "
            f"num_fragments={self.num_frags}, "
            f"total_atoms={self.frag_x.shape[0]}, "
            f"total_edges={self.frag_edge_index.shape[1]}"
        )

    def encode_all_fragments(self):
        """
        使用当前JK-GIN参数编码全部BRICS片段。

        返回：
            all_frag_embeddings:
                [num_fragments, output_dim]
        """

        # ====================================================
        # 第一层GIN节点表示
        # ====================================================

        h1 = self.gin1(
            self.frag_x,
            self.frag_edge_index
        )

        h1 = F.relu(h1)
        h1 = self.dropout(h1)

        # ====================================================
        # 第二层GIN节点表示
        # ====================================================

        h2 = self.gin2(
            h1,
            self.frag_edge_index
        )

        h2 = F.relu(h2)
        h2 = self.dropout(h2)

        # ====================================================
        # 对每个片段图分别进行全局平均池化
        # ====================================================
        # frag_batch_index记录每个原子属于哪个片段图。
        #
        # h1_pool: [num_fragments, hidden_dim]
        # h2_pool: [num_fragments, hidden_dim]
        # ====================================================

        h1_pool = global_mean_pool(
            h1,
            self.frag_batch_index,
            size=self.num_frags
        )

        h2_pool = global_mean_pool(
            h2,
            self.frag_batch_index,
            size=self.num_frags
        )

        # ====================================================
        # Jumping Knowledge
        # ====================================================
        # 同时保留第一层和第二层GIN信息
        # ====================================================

        h_jk = torch.cat(
            [h1_pool, h2_pool],
            dim=-1
        )

        all_frag_embeddings = self.proj(h_jk)

        return all_frag_embeddings

    def forward(self,fragment_indices,fragment_idfs,fragment_mask):
        all_frag_embeddings = self.encode_all_fragments()
        frag_tokens = all_frag_embeddings[fragment_indices]

        frag_tokens = (frag_tokens* fragment_mask.unsqueeze(-1).float())

        # IDF只用于药物级片段表示聚合
        weights = (
                fragment_idfs.float()
                * fragment_mask.float()
        )

        weighted_sum = (
                frag_tokens
                * weights.unsqueeze(-1)
        ).sum(dim=1)

        weight_sum = weights.sum(
            dim=1,
            keepdim=True
        ).clamp(min=1e-8)

        h_frag_pooled = weighted_sum / weight_sum

        return frag_tokens, h_frag_pooled



class Model_F1_C_S2_new(nn.Module):
    """F1 (JK) + C (MultiQuery) + S2 (Dense)"""
    def __init__(self, args, frag_graphs, node_feat_dim=6, edge_feat_dim=5,
                 frag_hidden_dim=128, frag_output_dim=256,
                 cross_hidden_dim=256, cross_heads=4, num_queries=4):
        super().__init__()

        self.semantic_encoder = SemanticEncoderV3_Dense(
            input_dim=args.smiles_input_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout
        )

        self.fragment_encoder = GINFragmentEncoder_JK1(
            frag_graphs=frag_graphs,
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=frag_hidden_dim,
            output_dim=frag_output_dim,
            dropout=args.dropout
        )

        self.cross_fusion = MultiQueryCrossAttentionFusion(
            sem_dim=1792,
            frag_dim=frag_output_dim,
            hidden_dim=cross_hidden_dim,
            num_heads=cross_heads,
            num_queries=num_queries,
            dropout=args.dropout
        )

        self.rgcn = R_GCN(
            num_feat=1792,
            num_hidden=args.hidden_dim,
            num_outputs=args.out_dim,
            dropout=args.dropout,
            rels=args.rels + 1
        )

        self.projection = nn.Sequential(
            nn.Linear(1792, 512), nn.ELU(), nn.Dropout(args.dropout),
            nn.Linear(512, args.out_dim), nn.LayerNorm(args.out_dim)
        )

        self.predictor = EnhancedPairPredictor(
            input_dim=args.out_dim, num_classes=args.rels, dropout=args.dropout
        )

        self.sem_cons_proj = nn.Sequential(nn.Linear(1792, 512), nn.ELU(), nn.Linear(512, 256))
        self.graph_cons_proj = nn.Sequential(nn.Linear(1792, 512), nn.ELU(), nn.Linear(512, 256))
        self.frag_cons_proj = nn.Sequential(nn.Linear(frag_output_dim, 512), nn.ELU(), nn.Linear(512, 256))

        self.temperature = args.temperature

    def forward(self, unified_edge_index, unified_edge_type, smiles_matrix,
                fragment_indices, fragment_idfs, fragment_mask, triples):

        h_sem, _ = self.semantic_encoder(smiles_matrix)
        # frag_tokens, h_frag_pooled = self.fragment_encoder(fragment_indices, fragment_mask)
        frag_tokens, h_frag_pooled = self.fragment_encoder(fragment_indices,fragment_idfs,fragment_mask)

        # h_fused, _ = self.cross_fusion(h_sem, frag_tokens, fragment_mask)
        h_fused, attn_weights = self.cross_fusion(h_sem, frag_tokens, fragment_mask)

        h_graph = self.rgcn(h_fused, unified_edge_index, unified_edge_type)
        node_emb_proj = self.projection(h_graph)

        z_sem = self.sem_cons_proj(h_sem)
        z_graph = self.graph_cons_proj(h_graph)
        z_frag = self.frag_cons_proj(h_frag_pooled)

        triples = triples.long()

        # h = node_emb_proj[triples[:, 0]]
        # t = node_emb_proj[triples[:, 1]]
        # output = self.predictor(h, t)
        #
        # return output, (z_sem, z_graph, z_frag)
        h = node_emb_proj[triples[:, 0]]
        t = node_emb_proj[triples[:, 1]]

        pair_embedding = torch.cat([h, t, torch.abs(h - t), h * t], dim=-1)

        output = self.predictor(h, t)

        return output, (z_sem, z_graph, z_frag, pair_embedding, attn_weights)









# ============================================================
# 消融实验公共组件
# ============================================================

class AblationBase(nn.Module):
    def _predict(self, node_emb, triples):
        triples = triples.long()
        return self.predictor(node_emb[triples[:, 0]], node_emb[triples[:, 1]])


# ============================================================
# 1. w/o Fragment
# 删除：JK-GIN、Cross-Attention、Fragment-Graph CL
# 保留：MolFormer、增强拓扑、R-GCN、Semantic-Graph CL
# ============================================================

class Model_WO_Fragment(AblationBase):
    def __init__(self, args, frag_graphs=None, node_feat_dim=6, edge_feat_dim=5, frag_hidden_dim=128, frag_output_dim=256, cross_hidden_dim=256, cross_heads=4, num_queries=4):
        super().__init__()
        self.semantic_encoder = SemanticEncoderV3_Dense(input_dim=args.smiles_input_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
        self.rgcn = R_GCN(num_feat=1792, num_hidden=args.hidden_dim, num_outputs=args.out_dim, dropout=args.dropout, rels=args.rels + 1)
        self.projection = nn.Sequential(nn.Linear(1792, 512), nn.ELU(), nn.Dropout(args.dropout), nn.Linear(512, args.out_dim), nn.LayerNorm(args.out_dim))
        self.predictor = EnhancedPairPredictor(input_dim=args.out_dim, num_classes=args.rels, dropout=args.dropout)
        self.sem_cons_proj = nn.Sequential(nn.Linear(1792, 512), nn.ELU(), nn.Linear(512, 256))
        self.graph_cons_proj = nn.Sequential(nn.Linear(1792, 512), nn.ELU(), nn.Linear(512, 256))

    def forward(self, edge_index, edge_type, smiles_matrix, fragment_indices, fragment_idfs, fragment_mask, triples):
        h_sem, _ = self.semantic_encoder(smiles_matrix)
        h_graph = self.rgcn(h_sem, edge_index, edge_type)
        node_emb = self.projection(h_graph)
        return self._predict(node_emb, triples), (self.sem_cons_proj(h_sem), self.graph_cons_proj(h_graph), None, None, None)


# ============================================================
# 2. w/o Cross-Attention
# 保留片段编码，用简单线性投影+残差相加替代Cross-Attention
# ============================================================

class Model_WO_CrossAttention(AblationBase):
    def __init__(self, args, frag_graphs, node_feat_dim=6, edge_feat_dim=5, frag_hidden_dim=128, frag_output_dim=256, cross_hidden_dim=256, cross_heads=4, num_queries=4):
        super().__init__()
        self.semantic_encoder = SemanticEncoderV3_Dense(input_dim=args.smiles_input_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
        self.fragment_encoder = GINFragmentEncoder_JK1(frag_graphs=frag_graphs, node_feat_dim=node_feat_dim, edge_feat_dim=edge_feat_dim, hidden_dim=frag_hidden_dim, output_dim=frag_output_dim, dropout=args.dropout)
        self.frag_fusion = nn.Sequential(nn.Linear(frag_output_dim, 1792), nn.LayerNorm(1792))
        self.rgcn = R_GCN(num_feat=1792, num_hidden=args.hidden_dim, num_outputs=args.out_dim, dropout=args.dropout, rels=args.rels + 1)
        self.projection = nn.Sequential(nn.Linear(1792, 512), nn.ELU(), nn.Dropout(args.dropout), nn.Linear(512, args.out_dim), nn.LayerNorm(args.out_dim))
        self.predictor = EnhancedPairPredictor(input_dim=args.out_dim, num_classes=args.rels, dropout=args.dropout)
        self.sem_cons_proj = nn.Sequential(nn.Linear(1792, 512), nn.ELU(), nn.Linear(512, 256))
        self.graph_cons_proj = nn.Sequential(nn.Linear(1792, 512), nn.ELU(), nn.Linear(512, 256))
        self.frag_cons_proj = nn.Sequential(nn.Linear(frag_output_dim, 512), nn.ELU(), nn.Linear(512, 256))

    def forward(self, edge_index, edge_type, smiles_matrix, fragment_indices, fragment_idfs, fragment_mask, triples):
        h_sem, _ = self.semantic_encoder(smiles_matrix)
        frag_tokens, h_frag = self.fragment_encoder(fragment_indices, fragment_idfs, fragment_mask)
        h_graph = self.rgcn(h_sem + self.frag_fusion(h_frag), edge_index, edge_type)
        node_emb = self.projection(h_graph)
        z_sem, z_graph, z_frag = self.sem_cons_proj(h_sem), self.graph_cons_proj(h_graph), self.frag_cons_proj(h_frag)
        return self._predict(node_emb, triples), (z_sem, z_graph, z_frag, None, None)


# ============================================================
# 3. w/o Topology Augmentation
# 模型结构与完整模型相同，main中改用original graph
# ============================================================

class Model_WO_TopologyAugmentation(Model_F1_C_S2_new):
    pass


# ============================================================
# 4. w/o Consistency Learning
# 模型结构与完整模型相同，main中只使用分类损失
# ============================================================

class Model_WO_ConsistencyLearning(Model_F1_C_S2_new):
    pass



class Model_WO_RGCN(AblationBase):
    def __init__(self,args,frag_graphs,node_feat_dim=6,edge_feat_dim=5,frag_hidden_dim=128,frag_output_dim=256,cross_hidden_dim=256,cross_heads=4,num_queries=4):
        super().__init__()
        self.semantic_encoder=SemanticEncoderV3_Dense(input_dim=args.smiles_input_dim,hidden_dim=args.hidden_dim,dropout=args.dropout)
        self.fragment_encoder=GINFragmentEncoder_JK1(frag_graphs=frag_graphs,node_feat_dim=node_feat_dim,edge_feat_dim=edge_feat_dim,hidden_dim=frag_hidden_dim,output_dim=frag_output_dim,dropout=args.dropout)
        self.cross_fusion=MultiQueryCrossAttentionFusion(sem_dim=1792,frag_dim=frag_output_dim,hidden_dim=cross_hidden_dim,num_heads=cross_heads,num_queries=num_queries,dropout=args.dropout)

        self.graph_mlp=nn.Sequential(
            nn.Linear(1792,1792),
            nn.LayerNorm(1792),
            nn.ELU(),
            nn.Dropout(args.dropout)
        )


        self.projection=nn.Sequential(nn.Linear(1792,512),nn.ELU(),nn.Dropout(args.dropout),nn.Linear(512,args.out_dim),nn.LayerNorm(args.out_dim))
        self.predictor=EnhancedPairPredictor(input_dim=args.out_dim,num_classes=args.rels,dropout=args.dropout)
        self.sem_cons_proj=nn.Sequential(nn.Linear(1792,512),nn.ELU(),nn.Linear(512,256))
        self.graph_cons_proj=nn.Sequential(nn.Linear(1792,512),nn.ELU(),nn.Linear(512,256))
        self.frag_cons_proj=nn.Sequential(nn.Linear(frag_output_dim,512),nn.ELU(),nn.Linear(512,256))


    def forward(self,edge_index,edge_type,smiles_matrix,fragment_indices,fragment_idfs,fragment_mask,triples):
        h_sem,_=self.semantic_encoder(smiles_matrix)
        frag_tokens,h_frag=self.fragment_encoder(fragment_indices,fragment_idfs,fragment_mask)
        h_fused,_=self.cross_fusion(h_sem,frag_tokens,fragment_mask)
        # 删除R-GCN
        h_graph=self.graph_mlp(h_fused)
        node_emb=self.projection(h_graph)
        z_sem=self.sem_cons_proj(h_sem)
        z_graph=self.graph_cons_proj(h_graph)
        z_frag=self.frag_cons_proj(h_frag)
        return self._predict(node_emb,triples),(z_sem,z_graph,z_frag, None, None)




class Model_WO_Pretrained(AblationBase):

    def __init__(self,args,frag_graphs,node_feat_dim=6,edge_feat_dim=5,frag_hidden_dim=128,frag_output_dim=256,cross_hidden_dim=256,cross_heads=4,num_queries=4):
        super().__init__()

        self.semantic_encoder=SemanticEncoderV3_Dense(
            input_dim=args.smiles_input_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout
        )

        self.fragment_encoder=GINFragmentEncoder_JK1(
            frag_graphs=frag_graphs,
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=frag_hidden_dim,
            output_dim=frag_output_dim,
            dropout=args.dropout
        )

        self.cross_fusion=MultiQueryCrossAttentionFusion(
            sem_dim=1792,
            frag_dim=frag_output_dim,
            hidden_dim=cross_hidden_dim,
            num_heads=cross_heads,
            num_queries=num_queries,
            dropout=args.dropout
        )

        self.rgcn=R_GCN(
            num_feat=1792,
            num_hidden=args.hidden_dim,
            num_outputs=args.out_dim,
            dropout=args.dropout,
            rels=args.rels+1
        )

        self.projection=nn.Sequential(
            nn.Linear(1792,512),
            nn.ELU(),
            nn.Dropout(args.dropout),
            nn.Linear(512,args.out_dim),
            nn.LayerNorm(args.out_dim)
        )

        self.predictor=EnhancedPairPredictor(
            input_dim=args.out_dim,
            num_classes=args.rels,
            dropout=args.dropout
        )

        self.sem_cons_proj=nn.Sequential(
            nn.Linear(1792,512),
            nn.ELU(),
            nn.Linear(512,256)
        )

        self.graph_cons_proj=nn.Sequential(
            nn.Linear(1792,512),
            nn.ELU(),
            nn.Linear(512,256)
        )

        self.frag_cons_proj=nn.Sequential(
            nn.Linear(frag_output_dim,512),
            nn.ELU(),
            nn.Linear(512,256)
        )


    def forward(self,edge_index,edge_type,smiles_matrix,fragment_indices,fragment_idfs,fragment_mask,triples):

        h_sem,_=self.semantic_encoder(smiles_matrix)

        frag_tokens,h_frag=self.fragment_encoder(
            fragment_indices,
            fragment_idfs,
            fragment_mask
        )

        h_fused,_=self.cross_fusion(
            h_sem,
            frag_tokens,
            fragment_mask
        )

        h_graph=self.rgcn(
            h_fused,
            edge_index,
            edge_type
        )

        node_emb=self.projection(h_graph)

        z_sem=self.sem_cons_proj(h_sem)
        z_graph=self.graph_cons_proj(h_graph)
        z_frag=self.frag_cons_proj(h_frag)

        return self._predict(node_emb,triples),(z_sem,z_graph,z_frag, None, None)



class Model_WO_MultiQuery(Model_F1_C_S2_new):

    def __init__(self,args,frag_graphs,node_feat_dim=6,edge_feat_dim=5,frag_hidden_dim=128,frag_output_dim=256,cross_hidden_dim=256,cross_heads=4,num_queries=4):

        super().__init__(
            args,
            frag_graphs,
            node_feat_dim,
            edge_feat_dim,
            frag_hidden_dim,
            frag_output_dim,
            cross_hidden_dim,
            cross_heads,
            num_queries
        )

        self.cross_fusion=MultiQueryCrossAttentionFusion(
            sem_dim=1792,
            frag_dim=frag_output_dim,
            hidden_dim=cross_hidden_dim,
            num_heads=cross_heads,
            num_queries=1,
            dropout=args.dropout
        )


# ============================================================
# Number of Queries parameter study
# Query=1 is Model_WO_MultiQuery and Query=4 is Model_F1_C_S2_new.
# These two variants complete the recommended {1, 2, 4, 8} setting.
# ============================================================

class Model_Query2(Model_F1_C_S2_new):
    """Full model with two cross-attention queries."""

    def __init__(self, args, frag_graphs, node_feat_dim=6, edge_feat_dim=5,
                 frag_hidden_dim=128, frag_output_dim=256,
                 cross_hidden_dim=256, cross_heads=4, num_queries=2):
        super().__init__(
            args=args,
            frag_graphs=frag_graphs,
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            frag_hidden_dim=frag_hidden_dim,
            frag_output_dim=frag_output_dim,
            cross_hidden_dim=cross_hidden_dim,
            cross_heads=cross_heads,
            num_queries=2
        )


class Model_Query8(Model_F1_C_S2_new):
    """Full model with eight cross-attention queries."""

    def __init__(self, args, frag_graphs, node_feat_dim=6, edge_feat_dim=5,
                 frag_hidden_dim=128, frag_output_dim=256,
                 cross_hidden_dim=256, cross_heads=4, num_queries=8):
        super().__init__(
            args=args,
            frag_graphs=frag_graphs,
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            frag_hidden_dim=frag_hidden_dim,
            frag_output_dim=frag_output_dim,
            cross_hidden_dim=cross_hidden_dim,
            cross_heads=cross_heads,
            num_queries=8
        )
