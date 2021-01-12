## HELLO WORLD

# Font: DejaVu Sans Mono


######################
#print(Hello World)

## Have to encode this as a string
print(Hello World)
print("Hello World")

## Data Types

# Strings (enclosed in parantheses)
print(type("Hello World"))

# Integers

print(type(7))

# Floats

print(type(7.0))

# Lists

print(type(["Hello World", "7", 7, 7.0]))

## Variable Assignemnt

str1 = "Hello World"
print(type(str1))

lst = [str1, "7", 7, 7.0]
print(type(lst))

## List indexing, list index starts at 0
# first element of the list is at lst[0]

print(lst[0])
print(type(lst[1]))


## Generally, operations within a type work how you'd expect

print("Hello "+"World")
print(7+7)
print(type(7+7))

print(7*7)

print(7.0+7.0)
print(type(7.0+7.0))

print([1,2,4] + [1])

# Sometimes operations don't preserve type
print(type(7/14))
print(type(14/7))

# // means integer division

print(type(14//7))
print(17//5)

## Some operations work between types

print(7*"Hello World! ")

print(7 + 7.0)
print(type(7 + 7.0))

print(7*7.0)

## But not always as you'd expect!
print(8*[1,2,3])

# To multiply each element of this list by 8, we can loop through and multiply

# First assign a variable to the list for easy reference
var1 = 8
lst = [1,2,3]
print(var1*lst)
# Write a for loop to go through each member of the lst and multiply it by 8
target = []
for element in lst:
    new = var1*element
    target.append(new)
    print(element, new, target)

lst2 = [2]
lst2.append(3)
print(lst2)

# Suppose I just want to do something 16 times
# range(n) takes in an integer n, and returns a list that is that is [0,1,..., n-1]



lst = list(range(16))
print(lst)

for element in range(16):
    print("Hello World!")

# List indexing

lst = ["Hello World", 20, "18", "ucla.edu"]
lst[0]
lst[3]


# With this in mind, return to the loop
numlst = [1,2,3]
var1 = 8

print(list(range(len(numlst))))
target=[]

for index in range(len(numlst)):
    element = numlst[index]
    new = var1*element
    target.append(new)
    print(index, element, new, target)





























