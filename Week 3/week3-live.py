#%% [Markdown]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %%
iris = sns.load_dataset('iris')
iris = pd.read_csv('iris.csv')
# %%
iris.head()
length = iris.sepal_length
width = iris.sepal_width
# %%
