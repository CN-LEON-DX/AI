input_data = input("your number: ")
print(input_data)
print(type(input_data))

# convert string to int

input_data = int(input_data)
print(input_data)
print(type(input_data))

#result:
# your number: 100
# 100
# <class 'str'>
# 100
# <class 'int'>


#-------------- Operators ---------
x = 1
y = 2
print(x / y)
print(x // y)
print(x ** y)

#res
# 0.5
# 0
# 1

def factorial(n):
    res = 1
    for i in range(1, n + 1):
        res *= i    
    return res

print(factorial(10))


arr = [4, 1, 5, 2, 3]

# func find the max value in array
print(max(arr))