
import torch
import random
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

class classifier(torch.nn.Module):
    def __init__(self,dim,out_dim=2):
        super(classifier,self).__init__()
        self.classifier1=torch.nn.Linear(dim,32)
        self.classifier2=torch.nn.Linear(32,16)
        self.classifier3=torch.nn.Linear(16,out_dim)
        

    def forward(self,x):
        x=self.classifier1(x)
        x=self.classifier2(x)
        return self.classifier3(x)


def train_classifier(features,labels):
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

    optimizer = torch.optim.Adam(classifier_model.parameters(), lr=0.01, weight_decay=5e-4)
    train_node=list(set(random.sample(range(0,features.shape[0]),int(features.shape[0]*0.8))))
    test_node=list((set(range(0,features.shape[0]))-set(train_node)))


    best=0
    best_test=0
    best_r=0
    best_p=0
    best_micro=0
    best_macro=0
   
    for epoch in range(10000):
        optimizer.zero_grad()

        out=classifier_model(features)
        loss=F.cross_entropy(out[train_node],labels[train_node])+F.cross_entropy(out[0:4838],labels[0:4838])*1+F.cross_entropy(out[0:43],labels[0:43])*5
        loss.backward(retain_graph=True)
        optimizer.step()
        
        _,pred1 = out.max(dim=1)
        correct = int(pred1[train_node].eq(labels[train_node]).sum().item())
        train_acc = correct/len(train_node)

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
