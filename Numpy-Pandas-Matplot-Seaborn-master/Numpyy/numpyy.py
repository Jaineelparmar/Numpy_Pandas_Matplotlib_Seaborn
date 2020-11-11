# What is NumPy?
# NumPy is a python library used for working with arrays.
# It also has functions for working in domain of linear algebra, fourier transform, and matrices.
# It is an open source project and you can use it freely.
# NumPy stands for Numerical Python.

# Why Use NumPy ?
# In Python we have lists that serve the purpose of arrays, but they are slow to process.
# NumPy aims to provide an array object that is up to 50x faster that traditional Python lists.
# The array object in NumPy is called ndarray, it provides a lot of supporting functions that make working with ndarray very easy.
# Arrays are very frequently used in data science, where speed and resources are very important.
# Ararys in Numpy are table of elements, all of same type

# -------------------------------------------------------------------------------------------------------------------
#CREATING ARRAY

#O-D ARRAY
import numpy as np
arr = np.array(42)
print(arr)

#1-D ARRAY
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(arr)

#2-D ARRAY
import numpy as np
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr)

# 3-D ARRAY
import numpy as np
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
print(arr)

#CHECK NUMBER OF DIMENSIONS
import numpy as np
a = np.array(42)
b = np.array([1, 2, 3, 4, 5])
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
print(a.ndim)
print(b.ndim)
print(c.ndim)
print(d.ndim)

#Higher Dimensional Arrays
import numpy as np
arr = np.array([1, 2, 3, 4], ndmin=5)
print(arr)
print('number of dimensions :', arr.ndim)

#INDEXING AND SLICING
import numpy as np
def arrays(arr):
    arr = np.array(arr, float)
    arr = arr[::-1]
    return arr
arr = input().strip().split(' ')
result = arrays(arr)
print(result)

import numpy as np
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(arr[0:2, 2])

import numpy as np
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(arr[0:2, 1:4])
# -------------------------------------------------------------------------------------------------------------------

#SHAPE AND RESHAPE
import numpy as np
n = np.array(list(map(int, input().split())))
print(n.reshape(3, 3))
print(np.reshape(n, (3, 3)))


#1-D TO 3-D
import numpy as np
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
newarr = arr.reshape(2, 3, 2)
print(newarr)

import numpy as np
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(arr.shape)


import numpy as np
arr = np.array([1, 2, 3, 4], ndmin=5)
print(arr)
print('shape of array :', arr.shape)


#RETURNS COPY OR VIEW?
import numpy as np
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
print(arr.reshape(2, 4).base)
print(arr.reshape(2, 4))


#UNKNOWN DIMENSION
import numpy as np
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
newarr = arr.reshape(2, 2, -1)
print(newarr)


# -------------------------------------------------------------------------------------------------------------------

#TRANSPOSE AND FLATTEN
import numpy as np
n, m = map(int, input().split())
a = np.array([list(map(int, input().split())) for _ in range(n)])
print(np.transpose(a))
print(a.flatten())

# -------------------------------------------------------------------------------------------------------------------

#CONCATENATE
import numpy as np
N, M, P = map(int, input().split())
x = np.array([list(map(int, input().split())) for i in range(N)])
y = np.array([list(map(int, input().split())) for i in range(M)])
print(np.concatenate((x, y), axis = 0))

# -------------------------------------------------------------------------------------------------------------------

#ZEROS AND ONES
import numpy as np
nums = tuple(map(int, input().split()))
print (np.zeros(nums, dtype = np.int))
print (np.ones(nums, dtype = np.int))

# -------------------------------------------------------------------------------------------------------------------

#EYE AND IDENTITY
import numpy as np
np.set_printoptions(legacy ='1.13')
N, M = map(int, input().split())
print(np.eye(N, M, k = 0))

# -------------------------------------------------------------------------------------------------------------------

#ARRAY MATHEMATICS
import numpy as np
N, M = map(int, input().split())
x = np.array([list(map(int,input().split())) for _ in range(N)])
y = np.array([list(map(int,input().split())) for _ in range(N)])
print(x+y)
print(x-y)
print(x*y)
print(x//y) #Floor division
print(x%y)  #remainder
print(x**y)

# -------------------------------------------------------------------------------------------------------------------

#FLOOR, CEIL, RINT
import numpy as np
# np.set_printoptions(sign=' ')
np.set_printoptions(legacy='1.13')
A = np.array(list(map(float, input().split())))
print(np.floor(A))
print(np.ceil(A))
print(np.rint(A))

# -------------------------------------------------------------------------------------------------------------------

#SUM AND PRODUCT
import numpy as np
n, m = map(int, input().split())
x = np.array([input().split() for _ in range(n)], int)
print(np.prod(np.sum(x, axis = 0), axis = 0))

# -------------------------------------------------------------------------------------------------------------------

#MIN AND MAX
import numpy as np
n, m = map(int, input().split())
x = [input().split() for i in range(n)]
arr = np.array(x, int)
print(np.max(np.min(arr, axis = 1)))

# -------------------------------------------------------------------------------------------------------------------

#MEAN, VAR AND STD
import numpy as np
np.set_printoptions(legacy = '1.13')
n, m = map(int, input().split())
# arr = np.array([input().split() for _ in range(n)], int)
arr = np.array([list(map(int, input().split())) for _ in range(N)])
print(np.mean(arr, axis=1))
print(np.var(arr, axis=0))
print(np.std(arr, axis=None))

# -------------------------------------------------------------------------------------------------------------------

#DOT AND CROSS
import numpy as np
N = int(input())
A = np.array([input().split() for _ in range(N)], int)
B = np.array([input().split() for _ in range(N)], int)
print(np.dot(A, B))

# -------------------------------------------------------------------------------------------------------------------

#INNER AND OUTER
import numpy as np
A = np.array(input().split(), int)
B = np.array(input().split(), int)
print(np.inner(A, B))
print(np.outer(A, B))

# -------------------------------------------------------------------------------------------------------------------

#POLYNOMIALS
import numpy as np
P = list(map(float, input().split()))
x = float(input())
print(np.polyval(P, x))             #The polyval tool evaluates the polynomial at specific value.
print(numpy.poly([-1, 1, 1, 10]) )   #The poly tool returns the coefficients of a polynomial with the given sequence of roots.
print(numpy.roots([1, 0, -1]) )      #The roots tool returns the roots of a polynomial with the given coefficients.
print(numpy.polyint([1, 1, 1]) )     #The polyint tool returns an antiderivative (indefinite integral) of a polynomial.
print(numpy.polyder([1, 1, 1, 1]) )  #The polyder tool returns the derivative of the specified order of a polynomial.
print(numpy.polyfit([0,1,-1, 2, -2], [0,1,1, 4, 4], 2))      
#The polyfit tool fits a polynomial of a specified order to a set of data using a least-squares approach.
#The functions polyadd, polysub, polymul, and polydiv 
#also handle proper addition, subtraction, multiplication, and division of polynomial coefficients, respectively.

# -------------------------------------------------------------------------------------------------------------------

#LINEAR ALGEBRA
import numpy as np
N = int(input())
A = np.array([input().split() for _ in range(N)], float)
A = np.array([list(map(float, input().split())) for _ in range(N)])
print(round(np.linalg.det(A), 2))   #The linalg.det tool computes the determinant of an array.

# The linalg.eig computes the eigenvalues and right eigenvectors of a square array.
vals, vecs = numpy.linalg.eig([[1 , 2], [2, 1]])
print(vals)                                    
print(vecs) 

print(numpy.linalg.inv([[1 , 2], [2, 1]]))  #The linalg.inv tool computes the (multiplicative) inverse of a matrix.

# -------------------------------------------------------------------------------------------------------------------

#DATA TYPES
# Data Types in Python
# By default Python have these data types:

# strings - used to represent text data, the text is given under quote marks. eg. "ABCD"
# integer - used to represent integer numbers. eg. -1, -2, -3
# float - used to represent real numbers. eg. 1.2, 42.42
# boolean - used to represent True or False.
# complex - used to represent a number in complex plain. eg. 1.0 + 2.0j, 1.5 + 2.5j
# Data Types in NumPy
# NumPy has some extra data types, and refer to data types with one character, like i for integers, u for unsigned integers etc.

# Below is a list of all data types in NumPy and the characters used to represent them.

# i - integer
# b - boolean
# u - unsigned integer
# f - float
# c - complex float
# m - timedelta
# M - datetime
# O - object
# S - string
# U - unicode string
# V - fixed chunk of memory for other type ( void )

import numpy as np
arr = np.array([1, 2, 3, 4])
print(arr.dtype)

import numpy as np
arr = np.array([1, 2, 3, 4], dtype='i4')
print(arr)
print(arr.dtype)

import numpy as np
arr = np.array(['apple', 'banana', 'cherry'])
print(arr.dtype)

import numpy as np
arr = np.array([1, 2, 3, 4], dtype='S')
print(arr)
print(arr.dtype)


import numpy as np
arr = np.array([1.1, 2.1, 3.1])
newarr = arr.astype('i')
print(newarr)
print(newarr.dtype)


import numpy as np
arr = np.array([1.1, 2.1, 3.1])
newarr = arr.astype(int)
print(newarr)
print(newarr.dtype)


import numpy as np
arr = np.array([1, 0, 3])
newarr = arr.astype(bool)
print(newarr)
print(newarr.dtype)


# -------------------------------------------------------------------------------------------------------------------

#COPY VS VIEW

# The Difference Between Copy and View
# The main difference between a copy and a view of an array is that the copy is a new array, and the view is just a view of the original array.
# The copy owns the data and any changes made to the copy will not affect original array, and any changes made to the original array will not affect the copy.
# The view does not own the data and any changes made to the view will affect the original array, and any changes made to the original array will affect the view.


import numpy as np
arr = np.array([1, 2, 3, 4, 5])
x = arr.copy()
arr[0] = 42
print(arr)
print(x)


import numpy as np
arr = np.array([1, 2, 3, 4, 5])
x = arr.view()
arr[0] = 42
print(arr)
print(x)


import numpy as np
arr = np.array([1, 2, 3, 4, 5])
x = arr.view()
x[0] = 31
print(arr)
print(x)


import numpy as np
arr = np.array([1, 2, 3, 4, 5])
x = arr.copy()
y = arr.view()
print(x.base)
print(y.base)


# -------------------------------------------------------------------------------------------------------------------

# NumPy Array Iterating
# Iterating Arrays
# Iterating means going through elements one by one.
# As we deal with multi-dimensional arrays in numpy, we can do this using basic for loop of python.
# If we iterate on a 1-D array it will go through each element one by one.


# Iterate on the elements of the following 1-D array:

import numpy as np
arr = np.array([1, 2, 3])
for x in arr:
  print(x)


# Iterating 2-D Arrays
# In a 2-D array it will go through all the rows.
# Iterate on the elements of the following 2-D array:

import numpy as np
arr = np.array([[1, 2, 3], [4, 5, 6]])
for x in arr:
  print(x)


# # Iterate on each scalar element of the 2-D array:

import numpy as np
arr = np.array([[1, 2, 3], [4, 5, 6]])
for x in arr:
  for y in x:
    print(y)


# # Iterate on the elements of the following 3-D array:

import numpy as np
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
for x in arr:
  print(x)


# # Iterate down to the scalars:

import numpy as np
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
for x in  np.nditer(arr):
    print(x)
for x in arr:
    for y in x:
        # print(y)
        for z in y:
            print(z)


# # Iterating Arrays Using nditer()
# # The function nditer() is a helping function that can be used from very basic to very advanced iterations. 
# It solves some basic issues which we face in iteration, lets go through it with examples.
# # Iterating on Each Scalar Element
# # In basic for loops, iterating through each scalar of an array we need to use n for loops which can be difficult to write for arrays with very high dimensionality.

# # Example
# # Iterate through the following 3-D array:

import numpy as np
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
for x in np.nditer(arr):
  print(x)


# # Iterating Array With Different Data Types
# # We can use op_dtypes argument and pass it the expected datatype to change the datatype of elements while iterating.
# # NumPy does not change the data type of the element in-place (where the element is in array) so it needs some other space to perform this action, that extra space is called buffer, and in order to enable it in nditer() we pass flags=['buffered'].

# # Example
# # Iterate through the array as a string:

import numpy as np
arr = np.array([1, 2, 3])
for x in np.nditer(arr, flags=['buffered'], op_dtypes=['S']):
  print(x)


# # Iterating With Different Step Size
# # We can use filtering and followed by iteration.

# # Example
# # Iterate through every scalar element of the 2D array skipping 1 element:

import numpy as np
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
for x in np.nditer(arr[:, ::2]):
  print(x)

# # Enumerated Iteration Using ndenumerate()
# # Enumeration means mentioning sequence number of somethings one by one.
# # Sometimes we require corresponding index of the element while iterating, the ndenumerate() method can be used for those usecases.

# # Example
# # Enumerate on following 1D arrays elements:

import numpy as np
arr = np.array([1, 2, 3])
for idx, x in np.ndenumerate(arr):
  print(idx, x)


# # Example
# # Enumerate on following 2D array's elements:

import numpy as np
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
for idx, x in np.ndenumerate(arr):
  print(idx, x)



# -------------------------------------------------------------------------------------------------------------------

#NumPy Joining Array
# Joining NumPy Arrays
# Joining means putting contents of two or more arrays in a single array.
# In SQL we join tables based on a key, whereas in NumPy we join arrays by axes.
# We pass a sequence of arrays that we want to join to the concatenate() function, along with the axis. If axis is not explicitly passed, it is taken as 0.
# “axis 0” represents rows and “axis 1” represents columns
# Example
# Join two arrays

import numpy as np
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.concatenate((arr1, arr2))
print(arr)


# Example
# Join two 2-D arrays along rows (axis=1):

import numpy as np
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])
arr = np.concatenate((arr1, arr2), axis=1)
print(arr) 


# # Joining Arrays Using Stack Functions
# # Stacking is same as concatenation, the only difference is that stacking is done along a new axis.
# # We can concatenate two 1-D arrays along the second axis which would result in putting them one over the other, ie. stacking.
# # We pass a sequence of arrays that we want to join to the concatenate() method along with the axis. If axis is not explicitly passed it is taken as 0.
# # Example

import numpy as np
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.stack((arr1, arr2), axis=0)
print(arr)


# # # Stacking Along Rows
# # # NumPy provides a helper function: hstack() to stack along rows.
# # # Example

import numpy as np
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.hstack((arr1, arr2))
print(arr)


# # # Stacking Along Columns
# # # NumPy provides a helper function: vstack()  to stack along columns. axis = 0
# # # Example

import numpy as np
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.vstack((arr1, arr2))
print(arr)

# # Stacking Along Height (depth)
# # NumPy provides a helper function: dstack() to stack along height, which is the same as depth. axis = 1
# # Example

import numpy as np
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.dstack((arr1, arr2))
print(arr)


# -------------------------------------------------------------------------------------------------------------------

#NumPy Splitting Array
# Splitting NumPy Arrays
# Splitting is reverse operation of Joining.
# Joining merges multiple arrays into one and Splitting breaks one array into multiple.
# We use array_split() for splitting arrays, we pass it the array we want to split and the number of splits.
# Example
# Split the array in 3 parts:

import numpy as np
arr = np.array([1, 2, 3, 4, 5, 6])
newarr = np.array_split(arr, 3)
print(newarr)
# Note: The return value is an array containing three arrays.

# If the array has less elements than required, it will adjust from the end accordingly.
# Example
# Split the array in 4 parts:

import numpy as np
arr = np.array([1, 2, 3, 4, 5, 6])
newarr = np.array_split(arr, 4)
print(newarr)
# Note: We also have the method split() available but it will not adjust the elements when elements are less in source array for splitting like in example above,
# array_split() worked properly but split() would fail.

# Split Into Arrays
# The return value of the array_split() method is an array containing each of the split as an array.
# If you split an array into 3 arrays, you can access them from the result just like any array element:
# Example
# Access the splitted arrays:

import numpy as np
arr = np.array([1, 2, 3, 4, 5, 6])
newarr = np.array_split(arr, 3)
print(newarr[0])
print(newarr[1])
print(newarr[2])

# Splitting 2-D Arrays
# Use the same syntax when splitting 2-D arrays.
# Use the array_split() method, pass in the array you want to split and the number of splits you want to do.
# Example
# Split the 2-D array into three 2-D arrays.

import numpy as np
arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
newarr = np.array_split(arr, 3)
print(*newarr)
# The example above returns three 2-D arrays.

# Let's look at another example, this time each element in the 2-D arrays contains 3 elements.
# Example
# Split the 2-D array into three 2-D arrays.

import numpy as np
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
newarr = np.array_split(arr, 3)
print(newarr)
# The example above returns three 2-D arrays.

# In addition, you can specify which axis you want to do the split around.
# The example below also returns three 2-D arrays, but they are split along the row (axis=1).
# Example
# Split the 2-D array into three 2-D arrays along rows.

import numpy as np
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
newarr = np.array_split(arr, 3, axis=1)
print(newarr)

# An alternate solution is using hsplit() opposite of hstack()
# Example
# Use the hsplit() method to split the 2-D array into three 2-D arrays along rows.

import numpy as np
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
newarr = np.hsplit(arr, 3)
print(newarr)
# Note: Similar alternates to vstack() and dstack() are available as vsplit() and dsplit().


# -------------------------------------------------------------------------------------------------------------------

#NumPy Searching Arrays
# Searching Arrays
# You can search an array for a certain value, and return the indexes that get a match.
# To search an array, use the where() method.
# Example
# Find the indexes where the value is 4:

import numpy as np
arr = np.array([1, 2, 3, 4, 5, 4, 4])
x = np.where(arr == 4)
print(x)
# The example above will return a tuple: (array([3, 5, 6],)
# Which means that the value 4 is present at index 3, 5, and 6.


# Example
# Find the indexes where the values are even:

import numpy as np
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
x = np.where(arr%2 == 0)
print(x)

# Example
# Find the indexes where the values are odd:

import numpy as np
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
x = np.where(arr%2 == 1)
print(x)


# Search Sorted
# There is a method called searchsorted() which performs a binary search in the array, 
# and returns the index where the specified value would be inserted to maintain the search order.
# The searchsorted() method is assumed to be used on sorted arrays.
# Example
# Find the indexes where the value 7 should be inserted:

import numpy as np
arr = np.array([6, 8, 7, 9])
x = np.searchsorted(arr, 7)
print(x)
# Example explained: The number 7 should be inserted on index 1 to remain the sort order.
# The method starts the search from the left and returns the first index where the number 7 is no longer larger than the next value.

# Search From the Right Side
# By default the left most index is returned, but we can give side='right' to return the right most index instead.
# Example
# Find the indexes where the value 7 should be inserted, starting from the right:

import numpy as np
arr = np.array([6, 7, 8, 9])
x = np.searchsorted(arr, 7, side='right')
print(x)
# Example explained: The number 7 should be inserted on index 2 to remain the sort order.
# The method starts the search from the right and returns the first index where the number 7 is no longer less than the next value.

# Multiple Values
# To search for more than one value, use an array with the specified values.
# Example
# Find the indexes where the values 2, 4, and 6 should be inserted:

import numpy as np
arr = np.array([1, 3, 5, 7])
x = np.searchsorted(arr, [2, 4, 6])
# x = np.searchsorted(arr, [0, 4, 6])
print(x)
# The return value is an array: [1 2 3] containing the three indexes where 2, 4, 6 would be inserted in the original array to maintain the order.



# # -------------------------------------------------------------------------------------------------------------------


# #Sorting Arrays
# Sorting means putting elements in a ordered sequence.
# Ordered sequence is any sequence that has an order corresponding to elements, like numeric or alphabetical, ascending or descending.
# The NumPy ndarray object has a function called sort(), that will sort a specified array.
# Example
# Sort the array:

import numpy as np
arr = np.array([3, 2, 0, 1])
print(np.sort(arr))
# Note: This method returns a copy of the array, leaving the original array unchanged.

# You can also sort arrays of strings, or any other data type:
# Example
# Sort the array alphabetically:

import numpy as np
arr = np.array(['banana', 'cherry', 'apple'])
print(np.sort(arr))

# Example
# Sort a boolean array:

import numpy as np
arr = np.array([True, False, True])
print(np.sort(arr))

# Sorting a 2-D Array
# If you use the sort() method on a 2-D array, both arrays will be sorted:
# Example
# Sort a 2-D array:

import numpy as np
arr = np.array([[3, 2, 4], [5, 0, 1]])
print(np.sort(arr))


# # -------------------------------------------------------------------------------------------------------------------


# #NumPy Filter Array
# Filtering Arrays
# Getting some elements out of an existing array and creating a new array out of them is called filtering.
# In NumPy, you filter an array using a boolean index list.
# A boolean index list is a list of booleans corresponding to indexes in the array.
# If the value at an index is True that element is contained in the filtered array, 
# if the value at that index is False that element is excluded from the filtered array.
# Example
# Create an array from the elements on index 0 and 2:

import numpy as np
arr = np.array([41, 42, 43, 44])
x = [True, False, True, False]
newarr = arr[x]
print(newarr)
# The example above will return [41, 43], why?
# Because the new filter contains only the values where the filter array had the value True, in this case, index 0 and 2.

# Creating the Filter Array
# In the example above we hard-coded the True and False values, but the common use is to create a filter array based on conditions.
# Example
# Create a filter array that will return only values higher than 42:

import numpy as np
arr = np.array([41, 42, 43, 44])
# Create an empty list
filter_arr = []
# go through each element in arr
for element in arr:
  # if the element is higher than 42, set the value to True, otherwise False:
  if element > 42:
    filter_arr.append(True)
  else:
    filter_arr.append(False)

newarr = arr[filter_arr]
print(filter_arr)
print(newarr)

# Example
# Create a filter array that will return only even elements from the original array:

import numpy as np
arr = np.array([1, 2, 3, 4, 5, 6, 7])
# Create an empty list
filter_arr = []
# go through each element in arr
for element in arr:
  # if the element is completely divisble by 2, set the value to True, otherwise False
  if element % 2 == 0:
    filter_arr.append(True)
  else:
    filter_arr.append(False)
newarr = arr[filter_arr]
print(filter_arr)
print(newarr)

# Creating Filter Directly From Array
# The above example is quite a common task in NumPy and NumPy provides a nice way to tackle it.
# We can directly substitute the array instead of the iterable variable in our condition and it will work just as we expect it to.
# Example
# Create a filter array that will return only values higher than 42:

import numpy as np
arr = np.array([41, 42, 43, 44])
filter_arr = arr > 42
newarr = arr[filter_arr]
print(filter_arr)
print(newarr)

# Example
# Create a filter array that will return only even elements from the original array:

import numpy as np
arr = np.array([1, 2, 3, 4, 5, 6, 7])
filter_arr = arr % 2 == 0
newarr = arr[filter_arr]
print(filter_arr)
print(newarr)


# # -------------------------------------------------------------------------------------------------------------------

# #Random Numbers in NumPy
# What is a Random Number?
# Random number does NOT mean a different number every time. Random means something that can not be predicted logically.
# Pseudo Random and True Random.
# Computers work on programs, and programs are definitive set of instructions. 
# So it means there must be some algorithm to generate a random number as well.
# If there is a program to generate random number it can be predicted, thus it is not truly random.
# Random numbers generated through a generation algorithm are called pseudo random.
# Can we make truly random numbers?
# Yes. In order to generate a truly random number on our computers we need to get the random data from some outside source. 
# This outside source is generally our keystrokes, mouse movements, data on network etc.
# We do not need truly random numbers, unless its related to security (e.g. encryption keys) or the basis of application is the randomness (e.g. Digital roulette wheels).
# In this tutorial we will be using pseudo random numbers.
# Generate Random Number
# NumPy offers the random module to work with random numbers.

# Example
# Generate a random integer from 0 to 100:
from numpy import random
x = random.randint(100)
print(x)

# Generate Random Float
# The random module's rand() method returns a random float between 0 and 1.
# Example
# Generate a random float from 0 to 1:
from numpy import random
x = random.rand()
print(x)

# Generate Random Array
# In NumPy we work with arrays, and you can use the two methods from the above examples to make random arrays.
# Integers
# The randint() method takes a size parameter where you can specify the shape of an array.

# Example
# Generate a 1-D array containing 5 random integers from 0 to 100:

from numpy import random
x=random.randint(100, size=(5))
print(x)

# Example
# Generate a 2-D array with 3 rows, each row containing 5 random integers from 0 to 100:

from numpy import random
x = random.randint(100, size=(3, 5))
print(x)

# Floats
# The rand() method also allows you to specify the shape of the array.
# Example
# Generate a 1-D array containing 5 random floats:

from numpy import random
x = random.rand(5)
print(x)

# Example
# Generate a 2-D array with 3 rows, each row containing 5 random numbers:

from numpy import random
x = random.rand(3, 5)
print(x)

# Generate Random Number From Array
# The choice() method allows you to generate a random value based on an array of values.
# The choice() method takes an array as a parameter and randomly returns one of the values.
# Example
# Return one of the values in an array:

from numpy import random
x = random.choice([3, 5, 7, 9])
print(x)
# The choice() method also allows you to return an array of values.

# Add a size parameter to specify the shape of the array.
# Example
# Generate a 2-D array that consists of the values in the array parameter (3, 5, 7, and 9):

from numpy import random
x = random.choice([3, 5, 7, 9], size=(3, 5))
print(x)


# # -------------------------------------------------------------------------------------------------------------------


# #NumPy ufuncs
# What are ufuncs?
# ufuncs stands for "Universal Functions" and they are NumPy functions that operates on the ndarray object.
# Why use ufuncs?
# ufuncs are used to implement vectorization in NumPy which is way faster than iterating over elements.
# They also provide broadcasting and additional methods like reduce, accumulate etc. that are very helpful for computation.
# ufuncs also take additional arguments, like:
# where boolean array or condition defining where the operations should take place.
# dtype defining the return type of elements.
# out output array where the return value should be copied.
# What is Vectorization?
# Converting iterative statements into a vector based operation is called vectorization.
# It is faster as modern CPUs are optimized for such operations.

# Add the Elements of Two Lists
# list 1: [1, 2, 3, 4]
# list 2: [4, 5, 6, 7]
# One way of doing it is to iterate over both of the lists and then sum each elements.
# Example
# Without ufunc, we can use Python's built-in zip() method:

x = [1, 2, 3, 4]
y = [4, 5, 6, 7]
z = []
for i, j in zip(x, y):
  z.append(i + j)
print(z)

# NumPy has a ufunc for this, called add(x, y) that will produce the same result.
# Example
# With ufunc, we can use the add() function:

import numpy as np
x = [1, 2, 3, 4]
y = [4, 5, 6, 7]
z = np.add(x, y)
print(z)