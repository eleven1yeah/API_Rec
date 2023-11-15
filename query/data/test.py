import torch
from torch_geometric.data import Data

# 创建节点特征矩阵
node_features = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)

# 创建节点标签
node_labels = torch.tensor([0, 1, 0], dtype=torch.long)

# 创建边索引
edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

# 创建边属性矩阵
edge_attr = torch.tensor([[0.5], [0.8]], dtype=torch.float32)

# 创建 train_data
train_data = Data(x=node_features, y=node_labels, edge_index=edge_index, edge_attr=edge_attr)

# 保存 train_data
torch.save(train_data, 'train_data.pt')
