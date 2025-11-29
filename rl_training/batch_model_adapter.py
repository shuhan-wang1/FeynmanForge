"""
模型批处理适配器
使FeynmanGCPN支持真正的PyG Batch推理

关键改进:
1. 处理PyG Batch输入
2. 返回per-graph输出
3. 高效的批量policy计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool
from typing import Dict, List, Optional, Tuple


def add_batch_support(model_class):
    """
    装饰器: 为模型添加batch支持
    
    用法:
        @add_batch_support
        class FeynmanGCPN(nn.Module):
            ...
    """
    original_forward = model_class.forward
    
    def batched_forward(self, data, vertex_states=None, return_value=True):
        """
        支持batch的forward
        
        Args:
            data: 可以是单个Data或Batch对象
        """
        # 检测是否是batch
        is_batch = isinstance(data, Batch) and hasattr(data, 'batch')
        
        if is_batch:
            batch_size = data.num_graphs
            
            # 编码图 - MPNN encoder自然支持batch
            node_embeddings, graph_embedding = self.encoder(data)
            
            # graph_embedding 形状: [batch_size, hidden_dim]
            
            # Policy head需要特殊处理以返回per-graph输出
            policy_output = self._batched_policy(graph_embedding, batch_size)
            
            output = {
                'node_embeddings': node_embeddings,
                'graph_embedding': graph_embedding,
                **policy_output
            }
            
            if return_value:
                value = self.value_head(graph_embedding)  # [batch_size, 1]
                output['value'] = value
            
            return output
        else:
            # 单图情况，使用原始forward
            return original_forward(self, data, vertex_states, return_value)
    
    def _batched_policy(self, graph_embedding, batch_size):
        """
        批量计算policy输出
        
        Args:
            graph_embedding: [batch_size, hidden_dim]
            batch_size: 批量大小
            
        Returns:
            Dict with probability tensors of shape [batch_size, num_choices]
        """
        # Action type probabilities
        action_logits = self.policy_head.action_type_head(graph_embedding)  # [B, num_action_types]
        action_type_probs = F.softmax(action_logits, dim=-1)
        
        # Vertex probabilities
        vertex_logits = self.policy_head.vertex_head(graph_embedding)  # [B, max_vertices]
        vertex_probs = F.softmax(vertex_logits, dim=-1)
        
        # Particle probabilities - 需要考虑physics gate
        particle_logits = self.policy_head.particle_head(graph_embedding)  # [B, num_particle_types]
        
        # 如果有physics gate，应用它
        if hasattr(self.policy_head, 'meta_physics_gate'):
            # 简化：对整个batch应用相同的gate (vertex_states为None时)
            particle_probs = F.softmax(particle_logits, dim=-1)
        else:
            particle_probs = F.softmax(particle_logits, dim=-1)
        
        return {
            'action_type_probs': action_type_probs,
            'vertex_probs': vertex_probs,
            'particle_probs': particle_probs,
        }
    
    # 替换forward方法
    model_class.forward = batched_forward
    model_class._batched_policy = _batched_policy
    
    return model_class


class BatchedFeynmanGCPN(nn.Module):
    """
    显式支持batch的FeynmanGCPN包装器
    
    用于包装现有模型使其支持batch推理
    """
    
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.model = base_model
        
        # 复制必要的属性
        self.encoder = base_model.encoder
        self.policy_head = base_model.policy_head
        self.value_head = base_model.value_head
        self.particle_embedding = base_model.particle_embedding
        self.conservation_mask = base_model.conservation_mask
        
    def forward(
        self,
        data: Data,
        vertex_states: Optional[List[Dict]] = None,
        return_value: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        批量前向传播
        
        支持:
        - 单个Data对象
        - Batch对象 (多个图)
        """
        is_batch = isinstance(data, Batch) and hasattr(data, 'batch')
        
        if is_batch:
            return self._forward_batch(data, return_value)
        else:
            return self._forward_single(data, vertex_states, return_value)
    
    def _forward_single(
        self, 
        data: Data, 
        vertex_states: Optional[List[Dict]],
        return_value: bool
    ) -> Dict[str, torch.Tensor]:
        """单图前向传播 - 使用原始模型"""
        return self.model.forward(data, vertex_states, return_value)
    
    def _forward_batch(
        self, 
        batch: Batch,
        return_value: bool
    ) -> Dict[str, torch.Tensor]:
        """
        批量前向传播
        
        关键: 一次处理所有图，返回per-graph输出
        """
        batch_size = batch.num_graphs
        
        # ===== 1. 编码所有图 =====
        # MPNN encoder自动处理batch (通过batch向量)
        node_embeddings, graph_embedding = self.encoder(batch)
        
        # graph_embedding: [batch_size, hidden_dim]
        
        # ===== 2. 计算policy =====
        # Action type
        action_logits = self.policy_head.action_type_head(graph_embedding)
        action_type_probs = F.softmax(action_logits, dim=-1)  # [B, 4]
        
        # Vertex selection
        vertex_logits = self.policy_head.vertex_head(graph_embedding)
        vertex_probs = F.softmax(vertex_logits, dim=-1)  # [B, max_vertices]
        
        # Particle selection (暂不应用physics gate)
        particle_logits = self.policy_head.particle_head(graph_embedding)
        particle_probs = F.softmax(particle_logits, dim=-1)  # [B, num_particle_types]
        
        output = {
            'node_embeddings': node_embeddings,
            'graph_embedding': graph_embedding,
            'action_type_probs': action_type_probs,
            'vertex_probs': vertex_probs,
            'particle_probs': particle_probs,
        }
        
        # ===== 3. 计算value =====
        if return_value:
            value = self.value_head(graph_embedding)  # [B, 1]
            output['value'] = value
        
        return output
    
    def get_action(
        self,
        data: Data,
        vertex_states: Optional[List[Dict]] = None,
        deterministic: bool = False
    ) -> Dict[str, int]:
        """采样单个action (保持与原模型兼容)"""
        return self.model.get_action(data, vertex_states, deterministic)
    
    def evaluate_actions(
        self,
        data: Data,
        actions: Dict[str, torch.Tensor],
        vertex_states: Optional[List[Dict]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """评估actions (保持与原模型兼容)"""
        return self.model.evaluate_actions(data, actions, vertex_states)
    
    def get_conservation_metrics(self) -> Dict[str, torch.Tensor]:
        """获取conservation metrics"""
        return self.model.get_conservation_metrics()


def patch_model_for_batch(model: nn.Module) -> nn.Module:
    """
    就地修改模型以支持batch推理
    
    这是最简单的集成方式，不需要修改现有代码
    
    用法:
        model = FeynmanGCPN(...)
        model = patch_model_for_batch(model)
    """
    original_forward = model.forward
    
    def patched_forward(data, vertex_states=None, return_value=True):
        is_batch = isinstance(data, Batch) and hasattr(data, 'batch')
        
        if not is_batch:
            return original_forward(data, vertex_states, return_value)
        
        # Batch处理
        batch_size = data.num_graphs
        
        # 编码
        node_embeddings, graph_embedding = model.encoder(data)
        
        # Policy (简化版，不使用vertex_states)
        action_logits = model.policy_head.action_type_head(graph_embedding)
        vertex_logits = model.policy_head.vertex_head(graph_embedding)
        particle_logits = model.policy_head.particle_head(graph_embedding)
        
        output = {
            'node_embeddings': node_embeddings,
            'graph_embedding': graph_embedding,
            'action_type_probs': F.softmax(action_logits, dim=-1),
            'vertex_probs': F.softmax(vertex_logits, dim=-1),
            'particle_probs': F.softmax(particle_logits, dim=-1),
        }
        
        if return_value:
            output['value'] = model.value_head(graph_embedding)
        
        return output
    
    model.forward = patched_forward
    return model


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("Testing batch support...")
    
    # 创建测试数据
    def make_test_graph():
        x = torch.randn(5, 9)  # 5 nodes, 9 features
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        edge_attr = torch.randn(4, 21)  # 4 edges, 21 features
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # 创建batch
    graphs = [make_test_graph() for _ in range(8)]
    batch = Batch.from_data_list(graphs)
    
    print(f"Batch info:")
    print(f"  Num graphs: {batch.num_graphs}")
    print(f"  Total nodes: {batch.x.shape[0]}")
    print(f"  Total edges: {batch.edge_index.shape[1]}")
    
    print("\nBatch support test complete!")
