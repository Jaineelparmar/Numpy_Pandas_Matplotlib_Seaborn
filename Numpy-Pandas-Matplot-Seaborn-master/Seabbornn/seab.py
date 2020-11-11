import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy import stats

#LINEPLOT
days = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
temperature = [36.6, 37, 37.7,39,40.1,43,43.4,45,45.6,40.1,44,45,46.8,47,47.8]
temp_df = pd.DataFrame({'days':days, 'temperature':temperature})
sns.lineplot(x = 'days', y = 'temperature', data = temp_df)
plt.show()

#RELPLOT
a = sns.load_dataset('flights')
sns.relplot(x='passengers', y='month', data=a)
plt.show()

b = sns.load_dataset('tips')
sns.relplot(x='time', y='tip', data=b)
plt.show()


#CATPLOT
b = sns.load_dataset('tips')
sns.catplot(x='day', y='total_bill', data=b)
sns.catplot(x='day', y='total_bill', data=b, kind='violin')
sns.catplot(x='day', y='total_bill', data=b, kind='boxen')
plt.show()


#DISTPLOT
c = np.random.normal(loc=5, size=100, scale=2)
sns.distplot(c)
plt.show()


#MAP = Apply a plotting function to each facet's subset of the data.
d = sns.load_dataset('iris')
print(d)
e = sns.FacetGrid(d, col='species')
e.map(plt.hist, 'sepal_length')
plt.show()


f = sns.load_dataset('flights')
g = sns.PairGrid(f)
g.map(plt.scatter)
plt.show()

sns.set(style='darkgrid') #can style it any
f = sns.load_dataset('flights')
g = sns.PairGrid(f)
g.map(plt.scatter)
plt.show()


#BOXPLOT
sns.set(style='white', color_codes=True) #can style it any
f = sns.load_dataset('tips')
sns.boxplot(x='day', y='total_bill', data=f)
plt.show()

sns.set(style='white', color_codes=True) #can style it any
f = sns.load_dataset('tips')
sns.boxplot(x='day', y='total_bill', data=f)
sns.despine(offset=10, trim=True)
plt.show()


#PALPLOT
c=sns.color_palette()
sns.palplot(c)
plt.show()

