#!/usr/bin/env python
# coding: utf-8

# # TP 2 Machine Learning
# 

# **import des libraries** 

# In[34]:


import numpy as np
import pandas as pd
import sqldf
import matplotlib.pyplot as plt
from datetime import datetime

import nltk
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import seaborn as sns


# **Chargement du Dataset**

# In[48]:


data = pd.read_csv('Retail.csv')
data


# **Data Cleaning**

# In[95]:


#Remplacement des valeurs vides par 0 pour les colonnes de type entier ou floar
data[['InvoiceNo', 'Quantity', 'UnitPrice', 'CustomerID']]=data[['InvoiceNo', 'Quantity', 'UnitPrice', 'CustomerID']].fillna(0)

# Mise à jour du format de la date
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], errors='coerce')
data['InvoiceDate']= data['InvoiceDate'].dt.strftime("%Y-%m-%d")

#Mise à jour du type de variable pour les colonnes de type int, float
data['CustomerID']=data['CustomerID'].astype(int)
data['UnitPrice'] = data['UnitPrice'].apply(lambda x: str.replace(',', '.'))
data['UnitPrice']=data['UnitPrice'].astype(float)

#Mettons en valeur absolue la colonne Quantity pour rendre positives les valeurs négatives
data['Quantity'] = data['Quantity'].abs()

#Ajout des colonnes year, month, day pour faciliter le filtre
data['year'] = pd.DatetimeIndex(data['InvoiceDate']).year
data['month'] = pd.DatetimeIndex(data['InvoiceDate']).month
data['day'] = pd.DatetimeIndex(data['InvoiceDate']).day

# Suppression des doublons dans la table
data = data.drop_duplicates()

#Observons le resultats de notre nettoyage
data.head()


# ## *Questions*
# 

# ### Question 1 - Volume

# #### Nombre de produits achetés lors d'une dépense 

# In[37]:


query_customer = """
    SELECT CustomerID, InvoiceNo, avg(Quantity) as avg_number_products
    FROM df
    group by CustomerID, InvoiceNo
"""

df_customer_mean = sqldf.run(query_customer)

df_customer_mean.head()


# In[38]:


disp = df_customer_mean.groupby('CustomerID').mean().reset_index()
disp.head()


# In[39]:


test = disp.head(10)
test.plot()


# #### Nombre médian et moyenne des produits achetés lors d'une dépense par pays

# In[52]:


volume = pd.DataFrame()
volume['average'] = data.groupby('Country')['Quantity'].mean()
volume['median'] = data.groupby('Country')['Quantity'].apply(np.median)
volume= volume.reset_index()

volume


# ##### Graphiques
# 

# In[41]:


volume.head(15).plot(kind='bar',x='Country', y='average')
volume.head(15).plot(kind='bar',x='Country', y='median')


# ### Question 2 - Montant

# #### Montant dépensé par un client

# In[42]:


query_customer_UnitPrice = """
    SELECT CustomerID, InvoiceNo, avg(UnitPrice*Quantity) as mean
    FROM df
    group by CustomerID, InvoiceNo
"""
df_customer_mean_price = sqldf.run(query_customer_UnitPrice)
df_customer_mean_price.head()


# In[43]:


disp = df_customer_mean_price.groupby('CustomerID').mean().reset_index()
disp.head()


# In[53]:


disp = disp.head()


# In[104]:


x = disp['CustomerID']
y = disp['mean']

fig, ax = plt.subplots()
ax.bar(x, y)

plt.show()


# ### Distribution

# #### Montant moyen et médian d'un panier client par pays

# In[55]:


data_test = df.copy(deep=True)
montant = pd.DataFrame()
data_test['cart_amount'] = data_test[['Quantity', 'UnitPrice']].apply(lambda x : (x['Quantity'] * x['UnitPrice']), 1)
montant['average'] = data_test.groupby('Country')['cart_amount'].mean()
montant['median'] = data_test.groupby('Country')['cart_amount'].apply(np.median)

montant


# #### Graphique

# In[105]:


fig, ax = plt.subplots(figsize=(30,30))
fig.tight_layout(pad=5)

def plot_hor_bar(subplot, data):
    plt.subplot(1,2,subplot)
    if subplot==1:
        ax = sns.barplot(y='Country', x='average', data=data,
                     color='green')
        plt.title("Montant moyen par pays",
          fontsize=30)
    else : 
        ax = sns.barplot(y='Country', x='median', data=data,
                     color='green')
        plt.title("Montant median par pays",
          fontsize=30)
    plt.xticks(fontsize=30)
    plt.ylabel(None)
    plt.yticks(fontsize=30)
    sns.despine(left=True)
    ax.grid(False)
    ax.tick_params(bottom=True, left=False)
    return None

plot_hor_bar(1, montant[['Country', 'average']])
plot_hor_bar(2, montant[['Country', 'median']])

plt.show()


# ### Question 3 - Volume

# #### Top 5 des produits les plus vendu le premier trimestre de 2011

# In[65]:


top_5_products = """
    SELECT StockCode, sum(Quantity) as total_quantity
    FROM data
    where year=2011 and month between 1 and 3
    group by StockCode 
    order by total_quantity desc
    limit 5
"""

df_top_5_products = sqldf.run(top_5_products)
df_top_5_products


# #### Top 5 des produits les plus vendu le deuxième trimestre de 2011

# In[66]:


top_5_products_second_trimestre = """
    SELECT StockCode, sum(Quantity) as total_quantity
    FROM data
    where year=2011 and month between 4 and 6
    group by StockCode 
    order by total_quantity desc
    limit 5
"""

df_top_5_products_second_trimestre = sqldf.run(top_5_products_second_trimestre)
df_top_5_products_second_trimestre


# ### Question 4 - Montant

# #### Les 5 pays générant le plus gros chiffre d'affaire lors du premier trimestre 2011

# In[67]:


CA_country = """
    SELECT Country, sum(Quantity * UnitPrice) as CA
    FROM data
    where year=2011 and month between 1 and 3
    group by Country 
    order by CA desc
"""

df_CA_country = sqldf.run(CA_country)
df_CA_country = df_CA_country.head(5)
df_CA_country


# ### Question 5 - Montant

# In[69]:


df = data[(data['Country'].isin(df_CA_country.Country)) & (data['year']==2011) & (data['month'].between(1,3))]

df['cart_amount'] = df[['Quantity', 'UnitPrice']].apply(lambda x : (x['Quantity'] * x['UnitPrice']), 1)

df.head()


# In[70]:


evol = pd.DataFrame()
evol['median'] = df.groupby(['Country', 'month'])['cart_amount'].apply(np.median)
evol['average'] = df.groupby(['Country', 'month'])['cart_amount'].mean()
evol = evol.reset_index()

evol_Australia = evol[evol['Country']=='Australia']

evol_EIRE = evol[evol['Country']=='EIRE']

evol_France = evol[evol['Country']=='France']

evol_Netherlands = evol[evol['Country']=='Netherlands']

evol_United_Kingdom = evol[evol['Country']=='United Kingdom']

evol


# ##### Graphique

# In[74]:


fig, ax = plt.subplots(figsize=(20,5))
x = np.arange(len(evol))
width = 0.5
plt.bar(x-0.2, evol['median'],
        width, color='tab:green', label='median')
plt.bar(x+0.2, evol['average'],
        width, color='red', label='average')
plt.title('montant moyen et median d\'un panier client par pays', fontsize=25)
plt.xlabel(None)
plt.xticks(evol.index, evol['Country'], fontsize=8)
plt.yticks(fontsize=17)
sns.despine(bottom=True)
ax.grid(False)
ax.tick_params(bottom=False, left=True)
plt.legend(frameon=False, fontsize=15)
plt.show()


# ### Question 6 - Fréquence

# In[75]:


united_kingdom = """
    SELECT CustomerID, sum(Quantity) quantity, sum(Quantity*UnitPrice) price
    FROM data
    where Country='United Kingdom'
    group by CustomerID 
    order by quantity desc
    limit 100
"""

united_kingdom = sqldf.run(united_kingdom)
united_kingdom


# In[76]:


print("les 100 plus gros clients des UK achètent en moyenne " + '\033[1m' + str(united_kingdom.quantity.mean()) + '\033[0m' + " produits "
     + "pour en moyenne " + '\033[1m' + str(int(united_kingdom.price.mean())) + "$")


# ## Modèle

# In[77]:


data_test = data.copy(deep=True)
data_test['cart_amount'] = data_test[['Quantity', 'UnitPrice']].apply(lambda x : (x['Quantity'] * x['UnitPrice']), 1)
data_model = data_test[['InvoiceDate', 'UnitPrice', 'Quantity', 'cart_amount', 'Country', 'year', 'month', 'day']]
data_model.head()


# In[78]:


data_model = data_model.groupby(['InvoiceDate', 'year', 'month', 'day']).agg({'cart_amount':'sum'}).reset_index()
data_model


# ### Training et score

# In[79]:


x = data_model[['year', 'month', 'day']]
y = data_model['cart_amount']

X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2)

print("X_train = " , X_train.shape)
print("X_test = " , X_test.shape)
print("Y_train = " , Y_train.shape)
print("Y_test = " , Y_test.shape)


# In[83]:


model = RandomForestRegressor()
rf = model.fit(X_train, Y_train)
predict = rf.predict(X_test)
print( "La précision du modèle est de " + str(rf.score(X_train, Y_train)))


# ### Prédiction CA

# In[85]:


model = RandomForestRegressor()
rf = model.fit(x, y)

# Afin d'obtenir les jours manquants du mois de décembre 2011 du 10 au 31 mais aussi la journée du 3
list_day = [i for i in range(10,32)]
list_day.append(3)

data_december = pd.DataFrame.from_dict({'year': [2011 for i in range(len(list_day))],
                'month': [12 for i in range(len(list_day))],
                'day': list_day
                })

data_december.sort_values('day')


# In[86]:


data_december_final = data_december.copy(deep=True)
data_december_final['cart_amount']=rf.predict(data_december)
data_december_final.head()

data_model_december = data_model[(data_model['year']==2011) & (data_model['month']==12)][['year', 'month', 'day','cart_amount']]
data_december_final = pd.concat([data_model_december, data_december_final], ignore_index=True)
data_december_final.sort_values('day')


# In[87]:


print("le CA en décembre 2011 sera de : " + '\033[1m' +  str(data_december_final.cart_amount.sum()))


# ### Prédiction Achat

# In[88]:


data_purchase = data[['InvoiceNo', 'CustomerID', 'InvoiceDate', 'UnitPrice', 'Quantity', 'Country', 'year', 'month', 'day']]
data_purchase = data_purchase.groupby(['Country', 'CustomerID', 'InvoiceDate', 'year', 'month', 'day']).agg({'InvoiceNo':'count'})
data_purchase = data_purchase.reset_index()
data_purchase = data_purchase.rename({'InvoiceNo': 'purchase_count'}, axis=1)

data_purchase


# In[89]:


x_puchase = data_purchase[['year', 'month', 'day']]
y_purchase = data_purchase['purchase_count']
rf_purchase = model.fit(x_puchase, y_purchase)

# Pour obtenir les jours manquants du mois de décembre 2011 du 10 au 31 mais aussi la journée du 3
list_day_purchas = [i for i in range(10,32)]
list_day_purchas.append(3)

december_purchas = pd.DataFrame.from_dict({'year': [2011 for i in range(len(list_day_purchas))],
                'month': [12 for i in range(len(list_day_purchas))],
                'day': list_day_purchas
                })

december_purchas.sort_values('day')


# In[90]:


december_purchas['purchase_count']=rf.predict(december_purchas)
december_purchas


# In[91]:


data_purchase_december = data_purchase[(data_purchase['year']==2011) & (data_purchase['month']==12 ) & (data_purchase['Country']=='France')][['CustomerID','year', 'month', 'day','purchase_count']]
final_data_purchase_december = pd.concat([data_purchase_december, december_purchas], ignore_index=True)
final_data_purchase_december.sort_values('day')


# In[93]:


print("Le nombre de clients ayant réaliser au moins un achat en France en décembre 2011 sera de : " +  str(final_data_purchase_december.count()))


# In[ ]:




