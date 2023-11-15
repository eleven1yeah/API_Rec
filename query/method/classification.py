import numpy as np

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
np.random.seed(None)


class MyGMM(object):
    def __init__(self, K=2):
        """
        高斯混合模型，用EM算法进行求解
        :param K: 超参数，分类类别

        涉及到的其它参数：
        :param N: 样本量
        :param D: 单个样本的维度
        :param alpha: 模型参数，高斯函数的系数，决定高斯函数的高度，维度（K）
        :param mu: 模型参数，高斯函数的均值，决定高斯函数的中型位置，维度（K,D）
        :param Sigma: 模型参数，高斯函数的方差矩阵，决定高斯函数的形状，维度（K,D,D）
        :param gamma: 模型隐变量，决定单个样本具体属于哪一个高斯分布，维度(N,K)
        """
        self.K = K
        self.params = {
            'alpha': None,
            'mu': None,
            'Sigma': None,
            'gamma': None
        }

        self.N = None
        self.D = None

    def __init_params(self):

        alpha = np.random.rand(self.K)
        alpha = alpha / np.sum(alpha)
        mu = np.random.rand(self.K, self.D)
        Sigma = np.array([np.identity(self.D) for _ in range(self.K)])

        gamma = np.random.rand(self.N, self.K)

        self.params = {
            'alpha': alpha,
            'mu': mu,
            'Sigma': Sigma,
            'gamma': gamma
        }

    def _gaussian_function(self, y_j, mu_k, Sigma_k):
        '''
        计算高纬度高斯函数
        :param y_j: 第j个观测值
        :param mu_k: 第k个mu值
        :param Sigma_k: 第k个Sigma值
        :return:
        '''

        n_1 = self.D * np.log(2 * np.pi)

        _, n_2 = np.linalg.slogdet(Sigma_k)

        n_3 = np.dot(np.dot((y_j - mu_k).T, np.linalg.inv(Sigma_k)), y_j - mu_k)

        return np.exp(-0.5 * (n_1 + n_2 + n_3))

    '''
    labels是已知的query的分类
    y为当前所需分类的query集合（n_query,feature_dim）
    '''
    def _E_step(self, y,labels):
        alpha = self.params['alpha']
        mu = self.params['mu']
        Sigma = self.params['Sigma']

        fake_nodes=[]
        for j in range(self.N):
            y_j = y[j]
            gamma_list = []
            for k in range(self.K):
                alpha_k = alpha[k]
                mu_k = mu[k]
                Sigma_k = Sigma[k]
                gamma_list.append(alpha_k * self._gaussian_function(y_j, mu_k, Sigma_k))
           
            self.params['gamma'][j, :] = np.array([v / np.sum(gamma_list) for v in gamma_list])
        max_index = np.argmax(self.params['gamma'], axis=1)

        for i, y_i, index in enumerate(zip(y, max_index)):
            if index!= labels[i]:
                fake_nodes.append(y_i)
        for fake_node in fake_nodes:
            y.insert(fake_node,y.shape[0],y,index=0)
        return y
        

    def _M_step(self, y,label):
        mu = self.params['mu']
        gamma = self.params['gamma']
        for k in range(self.K):
            mu_k = mu[k]
            gamma_k = gamma[:, k]
            gamma_k_j_list = []
            mu_k_part_list = []
            Sigma_k_part_list = []
            for j in range(self.N):
                y_j = y[j]
                gamma_k_j = gamma_k[j]
                gamma_k_j_list.append(gamma_k_j)

                mu_k_part_list.append(gamma_k_j * y_j)

                Sigma_k_part_list.append(gamma_k_j * np.outer(y_j - mu_k, (y_j - mu_k).T))
            
            self.params['mu'][k] = np.sum(mu_k_part_list, axis=0) / np.sum(gamma_k_j_list)
            self.params['Sigma'][k] = np.sum(Sigma_k_part_list, axis=0) / np.sum(gamma_k_j_list)
            self.params['alpha'][k] = np.sum(gamma_k_j_list) / self.N

    "max_iter 调整 EM 算法的最大迭代次数"
    def fit(self, y, labels,max_iter=100):
        y = np.array(y)
        self.N, self.D = y.shape
        self.__init_params()

        for _ in range(max_iter):
            gen_y=self._E_step(y)
            self._M_step(gen_y,labels)

'用于生成样本数据集。'
'参数 n_ex 表示生成的样本数量，n_classes 表示类别数量，n_in 表示每个样本的特征维度，seed 是一个随机种子。' \
'该函数返回生成的样本数据 y。'
def get_samples(n_ex=1000, n_classes=3, n_in=2, seed=None):

    y, _ = make_blobs(
        n_samples=n_ex, centers=n_classes, n_features=n_in, random_state=seed)
    return y

'输入'
'n_classes 表示生成数据集的类别数量，labels 是一个已知的 query 分类的列表。'
'函数 run_my_model(n_classes, labels) 会调用 MyGMM 类的 fit 方法来拟合数据。' \
'它首先生成样本数据 y，然后通过调用 my.fit(y) 进行模型训练。' \
'在训练完成后，它根据模型的结果，将数据点按照聚类结果绘制在散点图上。'
def run_my_model(n_classes,labels):
    from matplotlib import pyplot as plt
    my = MyGMM()
    y = get_samples(n_classes=n_classes)
    print(y)
    my.fit(y)

    max_index = np.argmax(my.params['gamma'], axis=1)
    print(max_index)
    k_list=[]
    for i in range(n_classes):
        k_list.append([])

    for y_i, index in zip(y, max_index):
        k_list[index].append(y_i)

    for i in range(len(k_list)):
        k_list[i]= np.array(k_list[i])

    plt.scatter(k_list[0][:, 0], k_list[1][:, 1], c='red')
    plt.scatter(k_list[0][:, 0], k_list[1][:, 1], c='blue')
    #plt.scatter(k3_list[:, 0], k3_list[:, 1], c='green')
    plt.show()

"""y = get_samples(n_ex=1000, n_classes=3, n_in=2, seed=None)
labels = [0, 1, 0, 1, ...]
run_my_model(n_classes=3, labels=labels)"""

"""'输出结果包括：
y:生成的样本数据集，形状为 (n_samples, n_features)
max_index：根据模型预测的样本聚类结果，是一个长度为 n_samples 的列表，表示每个样本所属的聚类类别。
绘制的散点图将不同类别的样本以不同颜色区分，并展示在二维坐标系中。
但需要注意的是，代码中在绘制散点图时，只针对了两个类别进行了处理，
所以 run_my_model() 函数只适用于 n_classes=2 的情况。
如需处理更多类别，可以相应地调整代码以适应更多类别的展示。"""