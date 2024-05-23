import numpy as np

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
np.random.seed(None)


class MyGMM(object):
    def __init__(self, K=2):
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

        n_1 = self.D * np.log(2 * np.pi)

        _, n_2 = np.linalg.slogdet(Sigma_k)

        n_3 = np.dot(np.dot((y_j - mu_k).T, np.linalg.inv(Sigma_k)), y_j - mu_k)

        return np.exp(-0.5 * (n_1 + n_2 + n_3))


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


    def fit(self, y, labels,max_iter=100):
        y = np.array(y)
        self.N, self.D = y.shape
        self.__init_params()

        for _ in range(max_iter):
            gen_y=self._E_step(y)
            self._M_step(gen_y,labels)

def get_samples(n_ex=1000, n_classes=3, n_in=2, seed=None):

    y, _ = make_blobs(
        n_samples=n_ex, centers=n_classes, n_features=n_in, random_state=seed)
    return y


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
