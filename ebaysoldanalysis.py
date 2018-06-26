import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
testset=pd.read_csv('ebaydata/TestSet.csv')
trainset = pd.read_csv('ebaydata/TrainingSet.csv')
#print(trainset.info())
train=trainset.drop(['EbayID','QuantitySold','SellerName','EndDay'],axis=1)
test=testset.drop(['EbayID','QuantitySold','SellerName','EndDay'],axis=1)
#train=trainset[['SellerClosePercent','HitCount','SellerSaleAvgPriceRatio','BestOffer']]
train_target=trainset['QuantitySold']
test_target=testset['QuantitySold']
n_trainSamples, n_features = train.shape
# 画出训练过程中SGDClassifier利用不同的mini_batch学习的效果
def plot_learning(clf,title):

    plt.figure(1)
    # 记录上一次训练结果在本次batch上的预测情况
    validationScore = []
    # 记录加上本次batch训练结果后的预测情况
    trainScore = []
    # 记录多次增量训练后的预测情况
    totalscore=[]
    # 最小训练批数
    mini_batch = 500

    for idx in range(int(np.ceil(n_trainSamples / mini_batch))):
        x_batch = train[idx * mini_batch: min((idx + 1) * mini_batch, n_trainSamples)]
        y_batch = train_target[idx * mini_batch: min((idx + 1) * mini_batch, n_trainSamples)]

        if idx > 0:
            validationScore.append(clf.score(x_batch, y_batch))
        clf.partial_fit(x_batch, y_batch, classes=range(5))
        totalscore.append(clf.score(train,train_target))
        if idx > 0:
            trainScore.append(clf.score(x_batch, y_batch))
    plt.plot(trainScore, label="train score")
    plt.plot(validationScore, label="validation socre")
    plt.plot(totalscore,label="total score")
    plt.xlabel("Mini_batch")
    plt.ylabel("Score")
    plt.legend(loc='best')
    plt.grid()
    plt.title(title)
    return totalscore

scaler = StandardScaler()
train = scaler.fit_transform(train)
test=scaler.fit_transform(test)
clf = SGDClassifier(penalty='l2', alpha=0.001)
plot_learning(clf,"SGDClassifier")
test_predict=clf.predict(test)
n_testsamples=test_target.size
j=0
for i in range(n_testsamples):
    if test_predict[i]==test_target[i]:
        j=j+1
print(j)
print(j/n_testsamples)
plt.show()
