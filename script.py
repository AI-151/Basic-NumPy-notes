import numpy

ThisIsAnArray = numpy.array([[1,2,3],
                             [4,5,6]
])

# The main type of data structure used in NumPy is the ndarray (N-dimensional Array), here is a 2x3 array
# A 2D Array is called a matrix
# An Array is similar to a List, however all the elements are the same type
# You can do maths directly onto it, e.g. use + - * or /

ExampleArray = numpy.array([[1,2],
                            [3,4],
                            [5,6]
])

print(ExampleArray.shape)   # Prints the dimensions of the Array i.e. 3x2
print(ExampleArray.size)    # Works similar to len(), returns the length of the whole Array, i.e. 6
print(ExampleArray.ndim)    # Prints the number of dimensions this Array has, i.e. 2

# Other Functions

print(numpy.zeros((2, 3)))
# Creates an Array of the dimensions specified in the tuple, and fills it with zeros

print(numpy.ones((2, 3)))
# Creates an Array of the dimensions specified in the tuple, and fills it with ones

print(numpy.full((2, 3), 7))
# Creates an Array of the dimensions specified in the tuple, and fills it with the specified value

print(numpy.eye(3))
# Creates an Identity Matrix of the form NxN, where N is the parameter
# An Identity Matrix has ones running along from the top left to bottom right corner (in a square array)
# Multiplying any matrix by the same dimension identity matrix results in the same matrix
# It is similar to multiplying any number by one

print(numpy.arange(0, 10, 2))
# Will essentially run a for i in range loop, where any values of i are added to a one-dimensional array
# In this case the array produced is [0,2,4,6,8]

print(numpy.linspace(0, 1, 5))
# Will create an array of length five starting from 0, ending with one, with evenly spaced values

print(numpy.random.rand(2, 3))
# Will create an array of specified dimensions, filing it in randomly with values of zero to one
# The probability of any given number appearing is the same - there is an even distribution

print(numpy.random.randn(2,3))
# Will create an array of random numbers 'around' zero, both positive and negative
# There is normal distribution, the probability of numbers appearing in the array is a bell curve
# This means numbers closer to zero are much more likely than those further away

print(numpy.random.randint(1, 10, size=(2, 3)))
# Creates an array of specified dimensions, and fills it in randomly with numbers within a given range

# Mathematical Functions

a = numpy.array([[1, 2, 3],[4,5,6]])
b = numpy.array([[4, 5, 6],[7,8,9]])

print(a + b)
# Normal mathematical functions can be applied to an array.

print(a * b)
# This returns the dot product of the two.

print(a ** 2)
# Another mathematical function - you can square the array.

print(numpy.sqrt(a))
# You can perform the square root function as well.

print(ExampleArray.transpose())
# Flips the dimensions of the array, this 3x2 array will become 2x3

print(numpy.dot(a,numpy.transpose(b)))
# Performs the dot product of two arrays, even for 2D matrices returns a matrix of the dot product
# Dimension 1 of matrix A needs to be equal to be equal to dimension 0 of B
# This means we need to transpose B to make A 2x3 and B 3x2, now they align

print(numpy.matmul(a,numpy.transpose(b)))
# Performs matrix multiplication
# Once again B needs to be transposed so the dimensions align

print(ExampleArray.reshape((6,1)))
# This function reshapes the array, changing the dimensions from 3x2 to 6x1

print(ExampleArray.mean())
# Prints the average value of the array

print(ExampleArray.sum())
# Prints out the sum of the array's values

print(numpy.where(ExampleArray>2,'Greater than two','Not greater than two'))
# .where() takes a condition and two values, what to return when true or false
# It will create a new array but replace the values with the true or false values

ExampleArray2 = numpy.array([[1,2],[3,4]])

print(numpy.linalg.inv(ExampleArray2))
# Returns the inverse matrix
# The inverse matrix multiplied by the original matrix will result in the identity matrix
# Not all matrices have an inverse - there are two conditions that a matrix must have to have an inverse
# The matrix must be a square
# The matrix must have a determinant of any value not 0

print(numpy.linalg.det(ExampleArray2))
# You can identify the determinant value of a matrix using this function

c = numpy.array([[1],
                 [2],
                 [3],
                 [4],
                 [5],
])

d = numpy.array([[6],
                 [7],
                 [8],
                 [9],
                 [10],
])



print(numpy.hstack((c,d)))
# Horizontally stacks together the two arrays

print(numpy.vstack((c,d)))
# Vertically stacks together the two arrays

# Features: size (m²) and bedrooms
Features_Array = numpy.array([
    [50, 1],
    [60, 2],
    [80, 2],
    [100, 3],
    [120, 4]
])

# Prices (in £k)
Target_Array = numpy.array([
    [150],
    [180],
    [220],
    [280],
    [350]
])

Features_Array_With_Bias = numpy.hstack((Features_Array, numpy.ones((Features_Array.shape[0], 1))))
# Creates a third column of 5 rows filled with ones, for the bias to be learned by the model
# By matrix multiplication, the bias is calculated as 1*b, so missing it out or any other value is wrong

weights = numpy.linalg.inv(Features_Array_With_Bias.T @ Features_Array_With_Bias) @ Features_Array_With_Bias.T @ Target_Array
new_house = numpy.array([[90, 3, 1]])
predicted_price = new_house @ weights
print("Predicted price (£k):", int(predicted_price[0][0].round(0)))