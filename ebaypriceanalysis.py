import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
testsubset = pd.read_csv('ebaydata/TestSubset.csv')
trainsubset = pd.read_csv('ebaydata/TrainingSubset.csv')
train=trainsubset.drop(['EbayID','Price','SellerName','EndDay'],axis=1)
test=testsubset.drop(['EbayID','Price','SellerName','EndDay'],axis=1)
train_price=trainsubset['Price']
test_price=testsubset['Price']
n_samples,n_feature=train.shape
minisamples=1000
scaler=StandardScaler()
train=scaler.fit_transform(train)
test=scaler.fit_transform(test)
crf=SGDRegressor()
trainscore=[]
vaidscore=[]
for i in range(int(np.ceil(n_samples/minisamples))):
    trainsample=train[i*minisamples:min((i+1)*minisamples,n_samples)]
    trainprice=train_price[i*minisamples:min((i+1)*minisamples,n_samples)]
    if i>0:
        vaidscore.append(crf.score(trainsample,trainprice))
    crf.partial_fit(trainsample,trainprice)
    if i>0:
        trainscore.append(crf.score(trainsample,trainprice))
plt.figure(1)
plt.plot(vaidscore,label="vaidscore")
plt.plot(trainscore,label="trainscore")
plt.xlabel("Mini_batch")
plt.ylabel("Score")
plt.legend(loc='best')
plt.grid()
plt.title(SGDRegressor)
testprice_predict=crf.predict(test)
score=crf.score(test,test_price)
print(score)
plt.figure(2)
plt.subplot(211)
plt.plot(test_price,label="test_price")
plt.xlabel("sample")
plt.ylabel("price")
plt.legend(loc='best')
plt.grid()
plt.title(SGDRegressor)
plt.subplot(212)
plt.plot(testprice_predict,label="testprice_predic")
plt.xlabel("sample")
plt.ylabel("price")
plt.legend(loc='best')
plt.grid()


plt.show()


