#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from node2vec import Node2Vec
import networkx as nx
import pandas as pd
import pickle

# Load dataset of omics data (protein-protein interaction data or compound-compound interaction data)
cc = pd.read_csv('chemical_chemical_interaction.csv')

# Build the no-directed graph
graph_cc = nx.Graph()

# -> add weighted edges
for j in range(len(cc['chemical1'])):
    graph_cc.add_edge(cc['chemical1'].iloc[j], cc['chemical2'].iloc[j], weight=cc['combined_score'].iloc[j])
    
    
# Precompute probabilities and generate walks
node2vec = Node2Vec(graph_cc)

# Embed
model = node2vec.fit()  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)

# Save the model of Node2vec
with open('modelcc.pickle', mode='wb') as f:
    pickle.dump(model, f)

