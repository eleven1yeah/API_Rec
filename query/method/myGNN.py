import torch
import math
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv

from torch_geometric.utils import add_self_loops,degree,softmax
from torch_geometric.datasets import Planetoid
import ssl
import torch.nn.functional as F
import random
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence
import torch.nn.functional as F

import json
import numpy as np

# 定义数据处理函数collate_fn：
# 该函数用于对输入的数据进行处理，主要包括获取数据的真实长度、进行填充和打包操作。
def collate_fn(data):
    #data.sort(key=lambda x: len(x), reverse=True)
    seq_len = [s.size(0) for s in data] # 获取数据真实的长度
    data = pad_sequence(data, batch_first=True)  #对样本进行填充，保证样本数据长度相同，默认填充值为0
    data = pack_padded_sequence(data, seq_len, batch_first=True,enforce_sorted=False) #压缩掉无效的填充值
    return data

#定义模型类Net：该类是整个图神经网络模型的主体，包含了两层GATConv层，用于图卷积操作。
class Net(torch.nn.Module):
    def __init__(self,in_num,out_num):  #in_num输入特征的维度；out_num输出类别的数量
        super(Net,self).__init__() #初始化
        self.gat1=GATConv(in_num,8,8,dropout=0.6) #输入特征维度in_num 输出特征维度8 头数8 丢弃率0.6
        self.gat2=GATConv(64,out_num,1,dropout=0.6)#输入特征维度64 输出特征维度out_num 头数1

    def forward(self,data):
        x,edge_index=data.x, data.edge_index #从data从提取节点特征x和边索引edge_index
        #print(x)
        x=self.gat1(x,edge_index)
        #print(x)
        x=self.gat2(x,edge_index)
        
        #print(x)
        #return F.log_softmax(x,dim=1) #激活函数，对输出进行 log-softmax 操作，将输出转换为概率分布
        return x

#定义图注意力层类GraphAttentionLayer：该类是自定义的图注意力层，用于实现图注意力机制。
class GraphAttentionLayer(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True, #是否将多个头的输出拼在一起
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2):
 
        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads

        ''' 计算每个注意力头的隐藏特征维度 n_hidden
         如果 is_concat 为 True，表示在多头注意力中将多个头的特征拼接起来作为输出特征，
         那么要求输出特征维度 out_features 必须能够被注意力头的数量 n_heads 整除。
         然后，将输出特征维度平均分配给每个注意力头，得到每个头的隐藏特征维度 n_hidden。
         如果 is_concat 为 False，表示在多头注意力中将多个头的特征取平均作为输出特征，
         这种情况下每个注意力头的隐藏特征维度 n_hidden 直接等于输出特征维度 
        '''
        if is_concat:
            assert out_features % n_heads == 0
            # If we are concatenating the multiple heads
            self.n_hidden = out_features // n_heads
        else:
            # If we are averaging the multiple heads
            self.n_hidden = out_features

        # 线性层，将输入特征映射到多头输出的隐藏层特征，其输出维度为 self.n_hidden * n_heads，不带偏置项
        self.linear = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        # 线性层计算注意力系数
        self.attn = nn.Linear(self.n_hidden * 2, 1, bias=False)
        # 激活函数，采用带有负斜率的 LeakyReLU 激活函数，默认负斜率为 0.2。
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        # Softmax 函数，计算注意力系数的归一化值，沿着维度 1 进行归一化。
        self.softmax = nn.Softmax(dim=1)
        # Dropout层，进行随机丢弃一部分注意力系数。
        self.dropout = nn.Dropout(dropout)

        self.a1=random.random() #随机数，用于在计算注意力系数时进行加权

    #三个参数分别为节点特征，邻接矩阵和边矩阵
    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor,edge_mat:torch.Tensor):
        print('GraphAttentionLayer')

        # Number of nodes
        n_nodes = h.shape[0]
        #计算节点特征经过线性映射后的多头特征 g，并调整其形状
        g = self.linear(h).view(n_nodes, self.n_heads, self.n_hidden)

        g_repeat = g.repeat(n_nodes, 1, 1)
       
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=0)
        #将 g_repeat_interleave 和 g_repeat 在最后一个维度上进行拼接，
        # 得到形状为 (n_nodes, n_nodes, n_heads, 2 * n_hidden) 的张量 g_concat。
        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)
       
        g_concat = g_concat.view(n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden)
        # 计算注意力系数 e
        e = self.activation(self.attn(g_concat))
        # squeeze函数去除最后一个维度，得到形状为 (n_nodes, n_nodes, n_heads) 的张量 e
        e = e.squeeze(-1)
        print('adj_mat.shape',adj_mat.shape)

        # 对 adj_mat 进行形状和值的断言检查，确保其符合预期的维度和形状
        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads

        #使用masked_fill方法将 e 和 edge_mat 中对应位置为 0 的元素替换为负无穷 -inf
        e = e.masked_fill(adj_mat == 0, float('-inf'))
        edge_mat=edge_mat.masked_fill(adj_mat == 0, float('-inf'))

        #将edge_mat与e进行线性组合，其中加权系数 self.a1 用于调整两者的权重。
        # 对加权后的结果应用 softmax 函数进行归一化，得到注意力系数 a
        a = self.softmax(edge_mat*self.a1+e*(1-self.a1))

        a = self.dropout(a)

        # 通过einsum函数计算加权的多头特征g的加权和，得到 attn_res，形状为 (n_nodes, n_heads, n_hidden)。
        # 输入a 三维数组，下标为ijh；输入g 三维数组，下标为jhf；输出attn_res 三维数组，下标为ihf
        attn_res = torch.einsum('ijh,jhf->ihf', a, g)

        # Concatenate the heads
        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        # Take the mean of the heads
        else:
            # 沿着第一个维度求均值，得到形状为 (n_nodes, n_hidden) 的张量并返回，表示平均多头输出
            return attn_res.mean(dim=1)

#定义自定义的GATConv类myGATConv：该类继承自MessagePassing类，实现了自定义的GATConv层，用于图卷积操作。
class myGATConv(MessagePassing):
    #in_channels 输入特征维数
    #negative_slope 采用leakyRELU的激活函数，x的负半平面斜率系数
    #add_self_loops GAT要求加入自环，即每个节点要与自身连接
    def __init__(self, in_channels,out_channels, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(myGATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads   #head即设置几个attention头
        self.concat = concat #concat用于设置多头输出时是否拼接
        #negative_slope设置leaklyRelu的参数
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
		#线性变换
        self.lin = torch.nn.Linear(in_channels, heads * out_channels, bias=False)
        #创建一个可学习的参数张量 att，用于存储多头注意力的权重
        self.att = torch.nn.parameter.Parameter(torch.Tensor(1, heads, out_channels))

        self.edge_mat=torch.Tensor() #创建一个新张量赋值给edge_mat,后续存储更新传递节点之间的边信息

        if bias and concat:
            #创建一个全零的参数张量，其形状为 (heads * out_channels)
            self.bias = torch.nn.Parameter(torch.zeros(heads * out_channels))
        elif bias and not concat:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None
		#用于重置参数
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.att)
        

    def forward(self, x, edge_index, edge_mat,return_attention_weights=None):
        self.edge_mat=edge_mat.repeat(1,self.heads).unsqueeze(-1)
      
        H, C = self.heads, self.out_channels

        x = self.lin(x).view(-1, H, C)
        #这里alpha的规模为[node_num,heads]
        alpha = (x * self.att).sum(dim=-1)

        if self.add_self_loops:
            num_nodes = x.size(0)
            num_nodes = x.size(0) if x is not None else num_nodes
            #edge_index, _ = remove_self_loops(edge_index)
            #edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        #print('yes')
        out = self.propagate(edge_index, x=x,
                             alpha=alpha)
        alpha = self._alpha
        self._alpha = None
        out=out+x
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            return out, (edge_index, alpha)

        else:
            return out


    #def message(self,x_j, x_i,alpha_j, index,edge_attr):
    def message(self, x_j, alpha_j, index):
        alpha = alpha_j
        #alpha_j[edge_num,heads]
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        #print('x_j',x_j.shape)
        #print('alpha_unsqueeze',alpha.unsqueeze(-1).shape)
        #print('edge_mat',self.edge_mat.shape)
        return x_j * (alpha.unsqueeze(-1)+self.edge_mat)
           
#focal loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#定义FocalLoss类：该类是实现了Focal Loss的损失函数，用于在分类任务中处理类别不平衡问题。
class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
 
    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)
 
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)
 
 
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
 
        probs = (P*class_mask).sum(1).view(-1,1)
 
        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)
 
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)
 
 
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

#计算每个类别在标签中的逆频率
#为解决数据集中类别不平衡问题，逆频率越大，权重越大，模型在训练过程中会更加关注该类别，以弥补样本数量的不足
def inverse_freq(label,class_num):
    res=[]
    num = label.shape[0]
    for i in range(class_num):
        res.append(num/(label[label==i].shape[0]))

    #return torch.tensor(res).cuda()
    return torch.tensor(res)


#TTAGN
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
#定义TTAGN类：该类是整个TTAGN模型的主体，包括GATConv层、GCNConv层、LSTM层和全连接层等组件，用于图分类任务。
class TTAGN(torch.nn.Module):
    def __init__(self,in_num,out_gat_num,out_gcn_num):
        super(TTAGN,self).__init__()
        #self.gat1=GATConv(in_num,8,16,dropout=0.6)
        #self.gat2=GATConv(8*16,out_gat_num,1,dropout=0.6)
        #self.gat1=GraphAttentionLayer(in_num,8,8,dropout=0.6)
        self.gat1=myGATConv(in_num,8,8,dropout=0.6)
        self.gat2=myGATConv(64,out_gat_num,1,dropout=0.6)

        self.conv1 = GCNConv(in_num, 64)
        self.conv2 = GCNConv(64, out_gcn_num)


        self.classifier1=torch.nn.Linear(in_num+out_gcn_num,32)
        self.classifier2=torch.nn.Linear(32,16)
        self.classifier3=torch.nn.Linear(16,4)
        #self.classifier=torch.nn.Linear(in_num+out_gcn_num,4)
        self.lstm_edge=torch.nn.LSTM(input_size=2,hidden_size=16,num_layers=1,bias=True,batch_first=True)
        self.lstm_node=torch.nn.LSTM(input_size=2,hidden_size=16,num_layers=1,bias=True,batch_first=True)
        self.dnn=torch.nn.Linear(self.lstm_edge.hidden_size,1)
        

    def forward(self,data,edge_data,adj_mat,node_trans=None):
        #print(4)
        #_,a=self.lstm(edge_data)
        #print(0)
        batch_x = collate_fn(edge_data)
        a,b=self.lstm_edge(batch_x)
        out_pad, out_len = pad_packed_sequence(a, batch_first=True)
        res=[]
        for  i,j in enumerate(out_len-1):
            res.append(out_pad[i,j,:].unsqueeze(0))
        res=torch.concat(res)
        #print('res',res)
        global trans_feature
        trans_feature=res

        c=self.dnn(res)

        edge_mat=c

        
        batch_x = collate_fn(node_trans)
        a,b=self.lstm_edge(batch_x)
        out_pad, out_len = pad_packed_sequence(a, batch_first=True)
        res=[]
        for  i,j in enumerate(out_len-1):
            res.append(out_pad[i,j,:].unsqueeze(0))
        res=torch.concat(res)

        '''for i,j,k in zip(data.edge_index[0],data.edge_index[1],c):
            #print(i,j)
            edge_mat[int(i)][int(j)]=k'''


        x,edge_index=data.x, data.edge_index
        #print(x.shape,res.shape)

        x=torch.concat([x,res],dim=-1)

        x_gat=self.gat1(x,edge_index,edge_mat)
        x_gat=self.gat2(x_gat,edge_index,edge_mat)

        #c=self.dnn(a[0]).squeeze(-1).squeeze(0)
        #print(1)
        #print(c)
        '''edge_mat=torch.zeros((data.x.shape[0],data.x.shape[0]),dtype=torch.float32)'''
        #print(data.x.shape[0])
        ''' for i,j,k in zip(data.edge_index[0],data.edge_index[1],c):
            #print(i,j)
            edge_mat[int(i)][int(j)]=k
        adj_mat=edge_mat.unsqueeze(-1).repeat(1,1,8)'''

        #print('x',x)
        
        #print(x)
        #print(torch.any(x.isnan()))
        #print(torch.any(x.isinf()))
        #x_gat=self.gat1(x,edge_index)
        #x_gat = F.leaky_relu(x_gat)
        #print(train_data.x[torch.isinf(train_data.x)])
        #print('x_gat',x_gat)
        #print(torch.any(x_gat.isnan()))
        #pos=torch.where(x_gat.isnan())
        #pos=torch.where(x_gat.isinf())
        #print(torch.any(x_gat.isnan()))
        #print(torch.any(x_gat.isinf()))
        #print(pos)
        #x_gat=self.gat2(x_gat,edge_index)
        #x_gat = F.leaky_relu(x_gat)
        x_gat = torch.where(torch.isinf(x_gat), torch.full_like(x_gat, 0), x_gat)
        x_gat = torch.where(torch.isnan(x_gat), torch.full_like(x_gat, 0), x_gat)
    
        #print('x_gat',x_gat)
        
        #print(x)
        #x_gat=self.gat2(x_gat,adj_mat,adj_mat)
        
        #print(x)
        #return F.log_softmax(x,dim=1)

        #GCN
        x_gcn = self.conv1(x, edge_index)
        x_gcn = F.leaky_relu(x_gcn)
        x_gcn = F.dropout(x_gcn, training=self.training)
        x_gcn = self.conv2(x_gcn, edge_index)
        x_gcn = F.leaky_relu(x_gcn)
        x_gcn = F.dropout(x_gcn, training=self.training)
        #print(torch.any(x_gcn.isnan()))
        #print(torch.any(x_gcn.isinf()))
        #print('x_gcn',x_gcn)


        x=torch.concat([x,x_gcn],dim=1)
        #x=torch.concat([x,x_gcn],dim=1)
        #print(torch.any(x.isnan()))
        #print(torch.any(x.isinf()))

        global features
        features=x

        
        #x=self.classifier(x)
        x=self.classifier1(x)
        x=F.leaky_relu(x)
        x=self.classifier2(x)
        x=F.leaky_relu(x)
        x=self.classifier3(x)
        #x = torch.sigmoid(x)
        #print(torch.any(x.isnan()))
        #print(torch.any(x.isinf()))
        #a=torch.where(x.isnan())
        #print(x[a[0][0]])
        #print(x1[a[0][0]])
        
        #print(a)
        #print('x',x)
        #print(torch.any(x_gat.isnan()),torch.any(x_gcn.isnan()),torch.any(x1.isnan()),torch.any(x.isnan()))

        return x,x_gcn,adj_mat


def train_GNN(edges_train=None, node_series=None):
    cuda = True if torch.cuda.is_available() else False#检查当前设备是否支持CUDA，如果支持，则使用CUDA进行加速。
    #cuda=False
    result=[]

    # 读取 JSON 文件
    with open('../data/graph_data.json', 'r', encoding='utf-8-sig') as file:
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
    #print(train_data)


    adj_mat=torch.zeros((train_data.x.shape[0],train_data.x.shape[0]),dtype=torch.float32)
    for i,j in zip(train_data.edge_index[0],train_data.edge_index[1]):
                #print(i,j)
                adj_mat[int(i)][int(j)]=1
    #device = torch.device( 'cpu')
    #train_data.to(device)
    #device = torch.device( 'cuda')
    model = TTAGN(train_data.x.shape[1]+16,32,2)
    preffix='_'
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    #edge_data = pad_sequence(edges,batch_first=True, padding_value=0)
    model.train()
    best=0
    best_test=0
    best_epoch=0
    best_p=0
    best_r=0
    best_micro=0
    best_macro=0
    focal_Loss=FocalLoss(4)

    edge_data=edges_train
    for i in range(len(edge_data)):
        if edge_data[i].shape[0]>100:
            edge_data[i]=edge_data[i][:100]
    for i in range(len(node_series)):
        if node_series[i].shape[0]>100:
            node_series[i]=node_series[i][:100]

    features=torch.Tensor()
    trans_feature=torch.Tensor()
    #生成训练节点和测试节点
    train_node=random.sample(range(0,train_data.x.shape[0]),int(train_data.x.shape[0]*0.8))
    test_node=list(set(range(0,train_data.x.shape[0]))-set(train_node))
    loss_list=[]
    if cuda:
        model=model.cuda()
        train_data=train_data.cuda()
        for i in range(len(edge_data)):
            edge_data[i]=edge_data[i].cuda()
        for i in range(len(node_series)):
            node_series[i]=node_series[i].cuda()
        adj_mat=adj_mat.cuda()
    else:
        model=model.cpu()
        train_data=train_data.cpu()
        for i in range(len(edge_data)):
            edge_data[i]=edge_data[i].cpu()
        for i in range(len(node_series)):
            node_series[i]=node_series[i].cpu()
        adj_mat=adj_mat.cuda()
    print('======start train=======')
    for epoch in range(1):
        optimizer.zero_grad()
        #model.to(device)
        #edge_data.to(device)
        #train_data.to(device)
        out,A_, A= model(train_data,edge_data,adj_mat,node_series)
        #print(out)
        #print('yes')

        
        #cross_entropy
        #loss = F.cross_entropy(out[train_node], train_data.y[train_node])

        #focal loss
        #loss=focal_Loss(out[train_node], train_data.y[train_node])

        #weighted cross_entropy
        #res=torch.mm(A_,A_.T)-A
        #loss1=torch.mm(res,res)/train_data.x.shape[0]
        #loss=F.cross_entropy(out[train_node], train_data.y[train_node])
        if cuda:
            loss=F.cross_entropy(out[train_node], train_data.y[train_node], weight=inverse_freq(train_data.y[train_node],4).cuda())
            #loss=F.cross_entropy(out[train_node], train_data.y[train_node])
        else:
            loss=F.cross_entropy(out[train_node], train_data.y[train_node], weight=inverse_freq(train_data.y[train_node],4))
            #loss=F.cross_entropy(out[train_node], train_data.y[train_node])
        loss_list.append(loss)
        loss.backward()
        optimizer.step()

        #print(loss)
        #print(loss)
        y=train_data.y
        #print(loss)
        _,pred1 = out.max(dim=1)

        #print(y[y==0].shape[0],y[y==1].shape[0],y[y==2].shape[0],y[y==3].shape[0])
        #print(pred1[pred1==0].shape[0],pred1[pred1==1].shape[0],pred1[pred1==2].shape[0],pred1[pred1==3].shape[0])
        correct = int(pred1[train_node].eq(train_data.y[train_node]).sum().item())
        train_acc = correct/len(train_node)
        if best<train_acc:
            best=train_acc
        if epoch%10==0:
            model.eval()
            pred,A_, A=model(train_data,edges_train,adj_mat,node_series)
            _,pred =pred.max(dim=1)
            #_,pred = model(train_data,edges_train).max(dim=1)
            #_, pred = model(train_data)
            correct = int(pred[test_node].eq(train_data.y[test_node]).sum().item())
            acc = correct/len(test_node)
            labels=train_data.y[test_node].cpu()
            pred=pred[test_node].cpu()
            p = precision_score(labels, pred, average="macro")
            r = recall_score(labels, pred, average="macro")
            micro_f = f1_score(labels, pred, average="micro")
            macro_f = f1_score(labels, pred, average="macro")
            if best_test<acc:
                best_test=acc
                best_epoch=epoch
                torch.save(model.cpu(),'best_'+preffix+'model.pth')
                torch.save(features.cpu(), preffix+'node_features.pt')
                torch.save(trans_feature.cpu(), preffix+'trans_features.pt')
                if cuda:
                    model.cuda()
                    features.cuda()
                    trans_feature.cuda()

            print('epoch:',epoch,'train_acc:', train_acc,'best acc: ',best)
            print('acc: ',acc,'best_test acc: ',best_test,'best_epoch: ',best_epoch)

            if best_r<r:
                best_r=r
                best_r=r
            if best_micro<micro_f:
                best_micro=micro_f
                best_micro=micro_f
            if best_macro<macro_f:
                best_macro=macro_f
                best_macro=macro_f
            if best_p<p:
                best_p=p
                best_p=p
            print('train')
            print(classification_report(train_data.y[train_node].cpu(), pred1[train_node].cpu()))
            #print(classification_report(train_data.y[train_node], pred1[train_node], target_names=['gambling','normal','phish-hack','ponzi']))
            print('test')
            print(classification_report(train_data.y[test_node].cpu(), pred.cpu()))


            result.append([epoch,train_acc,acc,p,r,micro_f,macro_f])
            print('epoch:',epoch,'train_acc:', train_acc,'best acc: ',best)
            print('acc: ',acc,'best_test acc: ',best_test,'best_epoch: ',best_epoch)

            print("Test: precision = {} recall = {} micro_f1 = {} macro_f = {}".format(p, r, micro_f, macro_f))
            print("best Test: precision = {} recall = {} micro_f1 = {} macro_f = {}".format(best_p, best_r, best_micro, best_macro))
            
            model.train()
    torch.save(model,preffix+'model.pth')
    print('train')
    print(classification_report(train_data.y[train_node].cpu(), pred1[train_node].cpu()))
    #print(classification_report(train_data.y[train_node], pred1[train_node], target_names=['gambling','normal','phish-hack','ponzi']))
    print('test')
    print(classification_report(train_data.y[test_node].cpu(), pred.cpu()))
    #print(classification_report(train_data.y[test_node], pred, target_names=['gambling','normal','phish-hack','ponzi']))

    return model

