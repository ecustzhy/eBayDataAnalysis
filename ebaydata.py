import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
testset=pd.read_csv('ebaydata/TestSet.csv')
trainset = pd.read_csv('ebaydata/TrainingSet.csv')
testsubset = pd.read_csv('ebaydata/TestSubset.csv')
trainsubset = pd.read_csv('ebaydata/TrainingSubset.csv')
train=trainset.drop(['EbayID','QuantitySold','SellerName','EndDay'],axis=1)
train_target=trainset['QuantitySold']
n_samples,n_features=train.shape
data=pd.DataFrame(np.hstack((train,train_target[:,None])),columns=list(range(n_features))+['issold'])
corr=data.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr)
plt.show()

