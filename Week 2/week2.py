# Week 2, Introduction to numpy

# Numpy is one of the most important data science libraries in python

import numpy as np

## In numpy we will work with arrays, which are similar to lists
## Numpy lists are optimized for data tasks and for effeciency
## First, we initialize an array

# We can initialize an array from a list
a = np.array(['First Element','Second Element','Third Element'])
print(a)
print(type(a))

lst = ['First Element','Second Element','Third Element']
aTest = np.array(lst)
print(a == aTest)

# Arrays are behave similarly to lists:

## We can loop over them:
for i in a:
    print(i)

## And they are indexed similarly
print(a[0])
print(a[1])
print(a[2])
print(a[3])

# But they have some properties that make then easier to work with then lists

a = np.array([1,3,23])
3*a














