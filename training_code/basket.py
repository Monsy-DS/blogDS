import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from apyori import apriori    
from joblib import Parallel, delayed
from multiprocessing import Manager

df = pd.read_csv('Groceries_dataset.csv')

df['Date']=pd.to_datetime(df['Date'])

one_hot = pd.get_dummies(df['itemDescription'])
df.drop('itemDescription', inplace=True, axis=1)
df = df.join(one_hot)

#transaction - if customer buy multiple product in one day - its a transaction

all_product = df.loc[:,'Instant food products':'zwieback'].sum(axis=1)
df['all_product']=all_product

records = df.groupby(['Member_number','Date']).sum()
records = records.reset_index()

all_prod = df.drop(['Member_number','Date','all_product'],axis=1)
df=df[['Member_number']]
all_prod = all_prod.columns

#replacing one for name
def get_Pnames(x):
    for product in all_prod:
        if x[product] >0:
            x[product] = product
    return x

records2 = records.apply(get_Pnames,axis=1)

x = records2.drop(['Member_number','Date','all_product'],axis=1).values
print(type(x))
x = [sub[~(sub==0)].tolist() for sub in x if sub[sub!=0].tolist()]
transaction=x

records2 = records2.drop(['Member_number','Date','all_product'],axis=1).values
y=records2

y = [sub[~(sub==0)].tolist() for sub in x if sub[sub!=0].tolist()]

rules = apriori(transaction, min_support=0.0001, min_confidance=0.01, 
                min_lift=2,min_lenght=2,target='rules')
association_result = list(rules)

hledam = 'ham'
List_k_ulozeni = []
for item in association_result:
    pair =item[0]
    items = [x for x in pair]
    #if items[0]==hledam:
     #   print(items[1])
    print(items)
    List_k_ulozeni.append(items)
    
x = 0
for items in List_k_ulozeni:
    if items[0]=='ham':
        print(items[1])
        x=1
if x==0:
    print('sorry, nic jsem nenasel')
    
manager = Manager()
temp = manager.list()

def func(v, temp):
    temp.append(v)
    return

_ = Parallel(n_jobs=4)(delayed(func)(v, temp) for v in range(10))

bow_name = 'bow_model.sav'
joblib.dump(bow2, bow_name)

import joblib
basket_values = 'basket.sav'
joblib.dump(List_k_ulozeni, basket_values)
