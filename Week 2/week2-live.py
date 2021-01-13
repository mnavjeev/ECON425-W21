# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Week 2: Introduction to Numpy and Pandas
# 
# Numpy and Pandas are some of the most important data science libraries in python. 
# 
# 

# %%
import numpy as np

# %% [markdown]
# In numpy we will work with numpy arrays, which are similar to lists. Numpy arrays are optimized for data tasks and for effeciency. First, we initialize an array

# %%
# We can initialize an array from a list
a = np.array(['First Element','Second Element','Third Element'])
print(a)
print(type(a))

lst = ['First Element','Second Element','Third Element']
aTest = np.array(lst)
print(a == aTest)

# %% [markdown]
# Numpy arrays can work similarly to lists. We can loop over them:

# %%
for i in a:
    print(i)

# %% [markdown]
# And they are indexed similarly

# %%
print(a[0])
print(a[1])
print(a[2])
print(a[3])

# %% [markdown]
# *But* they have some properties that make them easier to work with than lists for data science.
# 

# %%
# Multiplication works as we might like
first = np.array([1,2,3])
second = np.array([4,5,6])
print(3*first)
print(first*second)

# More generally, we can apply functions to the whole array
firstExp = np.exp(first)
print(firstExp)
reverse = np.log(firstExp)
print(reverse)

# %% [markdown]
# We can use numpy to tell us the shape of the array and to create multi-dimensional arrays. 

# %%
array2d = np.array([[1,2,3],[4,5,6]])
print(array2d)
print(array2d.shape)

# %% [markdown]
# As well as to create arrays of zeros and ones:
# 
# 

# %%
zeros = np.zeros(2)
print(zeros)
zeros2d = np.zeros([5,4])
print(zeros2d)

ones = np.ones(13)
print(ones)
ones2d = np.ones([2,6])
print(ones2d)

# %% [markdown]
# We can treat these arrays just like vectors and/or matrices. For any two real vectors $u = (u_1, \dots u_p)$ and $v = (v_1, \dots, v_p)$, their dot product is given:
# $$u \cdot v = \sum_{i=1}^p u_i v_i$$
# 
# So if $u = (1,-1,1)$ and $v = (2,5,3)$, we would expect 
# $$u\cdot v = 2 - 5 + 3  = 0$$
# (Note that if $u\cdot v = 0$, we say that $u$ and $v$ are *orthogonal*)
# 

# %%
u = np.array([1,-1,1])
v = np.array([2,5,3])
print(np.dot(u,v))

# %% [markdown]
# We can also treat multi-dimensional arrays as matrices. For example, for a square ($p \times p$) matrix, $X$, we may be interested in it's inverse (if it exits). The inverse of a matrix is a matrix $X^{-1}$ satisfying
# \begin{equation}
#     XX^{-1} = I_{p\times p}
# \end{equation}
# The matrix inverse exists and is unique whenever the determinant of a matrix is 0, $det(X)=0$. 
# 
# If $X$ is a $2 \times 2$ matrix, 
# \begin{equation*}
#     X = \begin{bmatrix} a & b \\ c & d \end{bmatrix}
# \end{equation*}
# then its determinant is given by 
# \begin{equation}
#     det(X) = \frac{1}{ad - bc}
# \end{equation}
# and, whenever the determinant is not 0, we can calculate the matrix inverse:
# \begin{equation}
#     X^{-1} = \frac{1}{ad - bc}\begin{bmatrix} d & -b \\ -c & a \end{bmatrix}
# \end{equation}
# So that (as an example):
# \begin{align*}
#     X = \begin{bmatrix} 4 & 7 \\ 2 & 6 \end{bmatrix} \implies X^{-1} &= \frac{1}{4\cdot 6 - 7\cdot 2}\begin{bmatrix}  6 & -7 \\ -2 & 4 \end{bmatrix} \\
#     X^{-1} &= \begin{bmatrix} 0.6 & -0.7 \\ -0.2 & 0.4 \end{bmatrix}
# \end{align*}
# 

# %%
X = np.array([[4 , 7], [2 , 6]]) 
print(X)


# %%
Xinv = np.linalg.inv(X)
print(Xinv)

# Test that this returns the identity matrix
test = np.dot(X,Xinv)
print(np.round(test, 2))

# %% [markdown]
# This sort of matrix inversion is especially useful when we are computing inverses of large matrices, as we have to do when we are conducting linear regression or fitting other machine learning models.  
# 
# 
# 
# 
# %% [markdown]
# ## Introduction to Pandas
# 
# Pandas builds off of numpy and is used to handle datasets. To find a specific dataset to use we are going to import one from the seaborn library. 
# 
# Once we read in a dataframe, we can use the '.head()' method to take look at the first few observations and see the variable names/features. 
# 

# %%
import pandas as pd
import seaborn as sns
# pd.read_csv("~path/to/data")
iris = sns.load_dataset('iris')
iris.head()

# %% [markdown]
# The '.count()' method will give us the number of non-empty entries in each column

# %%
print(iris.count())

# %% [markdown]
# You can access specific columns by indexing the data frame by the feature name

# %%
sepal_length = iris['sepal_length']
sepal_width = iris['sepal_width']
print(sepal_length)

# %% [markdown]
# 

