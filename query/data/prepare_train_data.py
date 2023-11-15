from py2neo import Graph,Node
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 连接到Neo4j数据库
uri = "bolt://localhost:7687"
username = "neo4j"
password = "Zhaoy000710@"
graph = Graph(uri, auth=(username, password))


# 自定义数据集类
class GraphDataset(Dataset):
    def __init__(self, triples):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        return torch.tensor(head), torch.tensor(relation), torch.tensor(tail)

# 从Neo4j数据库中获取实体和关联关系数据
# 查询实体属性特征
query = "MATCH (n) RETURN n.property_feature AS feature"
result = graph.run(query)
property_features = [record["feature"] for record in result]
property_features = np.array(property_features)

# 查询实体关联关系
query = "MATCH (n)-[r]->(m) RETURN ID(n) AS n_id, ID(m) AS m_id, ID(r) AS r_id"
result = graph.run(query)
entities = set()
relations = set()
entity_pairs = []
relation_ids = []



for record in result:
    n_id = record["n_id"]
    m_id = record["m_id"]
    r_id = record["r_id"]
    entities.add(n_id)
    entities.add(m_id)
    relations.add(r_id)
    entity_pairs.append((n_id, m_id))
    relation_ids.append(r_id)


# 构建关系三元组列表
triples = []

for n_id, m_id in entity_pairs:
    triples.append((n_id, relation_ids[n_id], m_id))

# 转换为自定义数据集
dataset = GraphDataset(triples)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 定义TransE模型
class TransEModel(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransEModel, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

    def forward(self, heads, relations, tails):
        head_embeddings = self.entity_embeddings(heads)
        relation_embeddings = self.relation_embeddings(relations)
        tail_embeddings = self.entity_embeddings(tails)

        return head_embeddings + relation_embeddings - tail_embeddings

# 创建TransE模型实例
num_entities = len(entities) + 1
num_relations = len(relations)
#print("Number of entities:", num_entities)
#print("Number of relations:", num_relations)

embedding_dim = 128
model = TransEModel(num_entities, num_relations, embedding_dim)

# 定义损失函数和优化器
criterion = nn.MarginRankingLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100

for epoch in range(num_epochs):
    for batch in dataloader:
        heads, relations, tails = batch
        optimizer.zero_grad()
        scores = model(heads, relations, tails)
        loss = criterion(scores, torch.zeros_like(scores))
        loss.backward()
        optimizer.step()

    print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, loss.item()))



'''import json
import numpy as np
# 读取 JSON 文件
with open('./graph_data.json', 'r', encoding='utf-8-sig') as file:
    data = json.load(file)

# 构建特征词汇表
feature_vocab = {'<PAD>': 0}  # 添加一个填充标记
relation_vocab = {'<PAD>': 0}  # 添加一个填充标记
feature_index = 1
relation_index = 1

# 存储 train_data 的字典
train_data = {
    'x': [],
    'y': [],
    'edges': [],
    'node_series': []
}

# 遍历每个子图
for graph in data:
    # 提取 "n", "r", "m" 组合
    n_node = graph['n']
    r_node = graph['r']
    m_node = graph['m']

    # 提取实体属性特征
    n_features = n_node['properties']['name']  # 假设实体属性特征为名称

    # 数值化实体属性特征
    n_features_encoded = []
    for feature in n_features:
        if feature not in feature_vocab:
            feature_vocab[feature] = feature_index
            feature_index += 1
        n_features_encoded.append(feature_vocab[feature])

    # 添加到 train_data.x
    train_data['x'].append(n_features_encoded)

    # 提取实体关联关系
    r_label = r_node['type']
    m_name = m_node['properties']['name']

    # 数值化实体关联关系
    r_label_encoded = []
    if r_label not in relation_vocab:
        relation_vocab[r_label] = relation_index
        relation_index += 1
    r_label_encoded.append(relation_vocab[r_label])

    # 添加到 train_data.y
    train_data['y'].append([n_features_encoded, r_label_encoded, m_name])

    # 提取实体间关联关系特征
    edge_features = np.array([])  # 假设实体间关联关系特征为空

    # 添加到 train_edges
    train_data['edges'].append(edge_features)

    # 提取实体节点的篇章说明
    node_series = "篇章说明"  # 假设实体节点的篇章说明为固定字符串

    # 添加到 node_series
    train_data['node_series'].append(node_series)

# 转换为 NumPy 数组
train_data['x'] = np.array(train_data['x'])
train_data['y'] = np.array(train_data['y'])
train_data['edges'] = np.array(train_data['edges'])

# 打印 train_data
print(train_data)

========================================================

# 提取相关信息
for graph in data:
    n_node = graph['n']
    n_identity = n_node['identity']
    n_labels = n_node['labels']
    n_name = n_node['properties']['name']

for graph in data:
    r_node = graph['r']
    r_identity = r_node['identity']
    r_start = r_node['start']
    r_end = r_node['end']
    r_type = r_node['type']
    r_name = r_node['properties']['name']

for graph in data:
    m_node = graph['m']
    m_identity = m_node['identity']
    m_labels = m_node['labels']
    m_name = m_node['properties']['name']


# 打印提取的信息
print("n_identity:", n_identity)
print("n_labels:", n_labels)
print("n_name:", n_name)

print("r_identity:", r_identity)
print("r_start:", r_start)
print("r_end:", r_end)
print("r_type:", r_type)
print("r_name:", r_name)

print("m_identity:", m_identity)
print("m_labels:", m_labels)
print("m_name:", m_name)
'''