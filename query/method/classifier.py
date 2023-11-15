#定义了一个名为classifier的PyTorch模型类和一个名为train_classifier的训练函数

'''输入：
    features：形状为(n, dim)的特征张量，其中n是样本数量，dim是每个样本的特征维度。
    labels：形状为(n,)的标签张量，包含每个样本的标签。
    输出：classifier_model：训练好的分类器模型。

这个文件的主要功能是训练一个多层感知机分类器(classifier)，并输出训练好的模型。
训练过程使用Adam优化器和交叉熵损失函数进行监督学习。
训练函数会在每个训练周期(epoch)计算训练准确率、测试准确率以及其他指标如精确率、召回率、F1分数等，
并输出评估结果。训练过程会持续进行多个周期，直到达到指定的训练周期数。
'''
import torch
import random
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

class classifier(torch.nn.Module):
    def __init__(self,dim,out_dim=2):
        super(classifier,self).__init__()
        #定义三个线性层
        self.classifier1=torch.nn.Linear(dim,32)
        self.classifier2=torch.nn.Linear(32,16)
        self.classifier3=torch.nn.Linear(16,out_dim)
        
    #定义了模型的前向传播过程。输入x经过classifier1、2、3层，然后返回模型的输出。
    def forward(self,x):
        x=self.classifier1(x)
        x=self.classifier2(x)
        return self.classifier3(x)

#train_classifier函数接收特征数据features和标签数据labels作为输入，并进行分类器的训练。
def train_classifier(features,labels):
    #函数内部首先判断是否可以使用CUDA加速，然后初始化分类器模型classifier_model。
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    num_gen=2000
    #preffix='yuanshu_model_3_'

    #print(labels[4745:4837])

    #generator=torch.load('./wgan/'+preffix+'generator.pth').to('cpu')

    classifier_model=classifier(features.shape[1],2)
    '''if cuda:
        generator=generator.cuda()
        classifier_model=classifier_model.cuda()
        features=features.cuda()
        labels=labels.cuda()'''

    if cuda:
        features=features.cuda()
        classifier_model=classifier_model.cuda()
        labels=labels.cuda()
    print('features',features.shape)
    print('labels',labels.shape)
    #print(features.device)
    #print(fake_features.device)
    '''feature_list=[]
    feature_list.append(features)
    for i in range(50):
        feature_list.append(features[:43])
    for i in range(40):
        feature_list.append(features[-50:])
    features=torch.concat(feature_list,dim=0)
    print(features.shape)'''

    #定义了一个Adam优化器(torch.optim.Adam)和训练集和测试集的索引train_node和test_node。
    optimizer = torch.optim.Adam(classifier_model.parameters(), lr=0.01, weight_decay=5e-4)
    train_node=list(set(random.sample(range(0,features.shape[0]),int(features.shape[0]*0.8))))
    test_node=list((set(range(0,features.shape[0]))-set(train_node)))


    best=0
    best_test=0
    best_r=0
    best_p=0
    best_micro=0
    best_macro=0
    #进入训练循环，进行多个epoch的训练。
    # 在每个epoch中，首先将模型参数的梯度清零(optimizer.zero_grad())，然后通过前向传播计算输出out。
    for epoch in range(10000):
        optimizer.zero_grad()

        out=classifier_model(features)
        #根据训练集的一部分计算损失函数loss，使用交叉熵损失函数(torch.nn.functional.cross_entropy)。
        # 其中，训练集部分的损失权重设为1，而其他部分（例如前43个样本和前4838个样本）的损失权重设为不同的值。
        loss=F.cross_entropy(out[train_node],labels[train_node])+F.cross_entropy(out[0:4838],labels[0:4838])*1+F.cross_entropy(out[0:43],labels[0:43])*5
        #调用反向传播和优化器的step函数更新模型的参数
        loss.backward(retain_graph=True)
        optimizer.step()
        
        _,pred1 = out.max(dim=1)
        correct = int(pred1[train_node].eq(labels[train_node]).sum().item())
        train_acc = correct/len(train_node)

        #在每个epoch的特定步骤，将模型设置为评估模式(classifier_model.eval())，进行测试集上的预测，
        #并计算精确度、召回率、F1分数和分类报告。同时，打印训练集和测试集上的性能指标。
        if best<train_acc:
            best=train_acc
        if epoch%100==0:
            classifier_model.eval()
            _,pred = classifier_model(features).max(dim=1)
            #_, pred = model(train_data)
            correct = int(pred[test_node].eq(labels[test_node]).sum().item())
            acc = correct/len(test_node)
            if best_test<acc:
                best_test=acc
                best_epoch=epoch
            #print('epoch:',epoch,'train_acc:', train_acc,'best acc: ',best)
            #print('acc: ',acc,'best_test acc: ',best_test,'best_epoch: ',best_epoch)
            labels_test=labels[test_node].cpu()
            pred=pred[test_node].cpu()
            p = precision_score(labels_test, pred, average="macro")
            r = recall_score(labels_test, pred, average="macro")
            micro_f = f1_score(labels_test, pred, average="micro")
            macro_f = f1_score(labels_test, pred, average="macro")
            #print("Test: precision = {} recall = {} micro_f1 = {} macro_f = {}".format(p, r, micro_f, macro_f))
            classifier_model.train()
            print('train')
            #print(classification_report(labels[train_node].cpu(), pred1[train_node].cpu(), target_names=['gambling','normal','phish-hack','ponzi']))
            print(classification_report(labels[train_node].cpu(), pred1[train_node].cpu()))
            print('test')
            #print(classification_report(labels[test_node].cpu(), pred.cpu(), target_names=['gambling','normal','phish-hack','ponzi']))
            print(classification_report(labels[test_node].cpu(), pred.cpu()))

            #追踪并更新最佳的性能指标值
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

            print('epoch:',epoch,'train_acc:', train_acc,'best acc: ',best)
            print('acc: ',acc,'best_test acc: ',best_test,'best_epoch: ',best_epoch)

            print("Test: precision = {} recall = {} micro_f1 = {} macro_f = {}".format(p, r, micro_f, macro_f))
            print("best Test: precision = {} recall = {} micro_f1 = {} macro_f = {}".format(best_p, best_r, best_micro, best_macro))

    return classifier_model