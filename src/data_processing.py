import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from itertools import combinations
from scipy.spatial.distance import jaccard

data = pd.read_excel('../data/dataset.xlsx')
data['Endometriosis'] = data['label']

data = data.drop(data.columns[0], axis=1) # remove first columns 

data['Depression / Anxiety'] = data[['Depression', 'Anxiety']].max(axis=1)
data = data.drop(['Depression', 'Anxiety'], axis=1)
data=data.drop(['row','label'],axis=1)
data['Painful Periods'] = data[['Menstrual pain (Dysmenorrhea)', 'Painful cramps during period']].max(axis=1)
data = data.drop(['Menstrual pain (Dysmenorrhea)', 'Painful cramps during period'], axis=1)

# Now for a better understanding of the data I turned these 
# numeric columns into boolean ones. This will make the plots of the 
# following cells easier to interpret at first glance
data_bool = data.copy()
for col in data.columns:
    data_bool[col] = data[col].apply(lambda x: True if x > 0 else False)

jaccard_matrix = pd.DataFrame(index=data_bool.columns, columns=data_bool.columns, dtype=float)
for pair in combinations(data_bool.columns, 2):
    jaccard_index = 1 - jaccard(data_bool[pair[0]], data_bool[pair[1]])
    jaccard_matrix.at[pair[0], pair[1]] = jaccard_index
    jaccard_matrix.at[pair[1], pair[0]] = jaccard_index

for col in data_bool.columns:
    jaccard_matrix.at[col, col] = 1

plt.figure(figsize=(10, 10))
plt.title('Jaccard Indices between Symptom Vectors')

# Create a heatmap
sns.heatmap(jaccard_matrix, annot=False, cmap=sns.color_palette("viridis", as_cmap=True),xticklabels=True,yticklabels=True)

plt.xticks(rotation=90,fontsize=6)
plt.yticks(rotation=0,fontsize=6)

plt.tight_layout()
path = os.path.join('../figures', 'jaccard_heatmap.svg')
plt.savefig(path)


_, axes = plt.subplots(nrows=5, ncols=3, figsize=(14, 30), sharey=True)
axes = [axis for subl in axes for axis in subl]
for col, ax in zip(data_bool.columns, axes):
    if col != 'Endometriosis':
        sns.countplot(x=col, hue="Endometriosis", data=data_bool, ax=ax, palette="Set2",legend=False)
    else:
        sns.countplot(x=col, data=data_bool, ax=ax, palette="Set2",legend=False)

plt.tight_layout()
path = os.path.join('../figures', 'symptoms.png')
plt.savefig(path)


