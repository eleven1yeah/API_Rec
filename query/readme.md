
For software configuration, all model are implemented in
- Python 3.7
- Pytorch-Geometric 2.0.3
- Pytorch 1.8.0
- Scikit-learn 0.24.1 封装机器学习常用算法
- CUDA 10.2

manage.py 用于运行样例回答界面
其中包含样例查询、回答、索引界面，用于演示

核心算法代码包含在method中

#classification.py
改良的GMM算法，用于分类regualr query 与 irregular query
labels是已知的query的分类
y为当前所需分类的query集合（n_query,feature_dim）

#classifier.py
用于判断与问询语句最为相关的子图
输入为featuers（n_subgraph,dim_query+dim_subgraph）
其中每行为中心子图嵌入特征与问句特征的拼接

#sentence_parse.py
解析句子的依存句法关系，用于regular query查询分析查询语句的内容

#triple_extraction.py
三元组抽取模块，用于依照语义角色抽取三元组中的实体关系

#TransE.py
TransE算法用于嵌入实体关系的特征，用与高相关实体的中心子图的特征计算

#myGNN
基于与文本、关系序列为注意力的图卷积神经网络，用于计算高相关节点邻接子图的特征
输出特征用于比较筛选最符合文本特征的相关回答的子图
train_data 为训练的知识图数据
train_data.x 实体对应的属性特征 （n_node,dim_feature）
train_data.y 为实体关联 (2,n_edges)
train_edges 为实体间的关联关系，与 train_data.y 对应(n_edges,n_realtion,dim_edge_feature) 
对于任意两实体之间可能存在不止一种的关联关系 n_realtion
node_series 为实体对应的篇章说明，可以选择地用于实体节点的嵌入