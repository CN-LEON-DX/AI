def max_kerner(num_list, k):
    res = []
    for i in range(len(num_list) - k):
        res.append(max(num_list[i:i+k]))
    return res

num_list = [3, 4, 5, 1, -44, 5, 10, 12, 33, 1]
# output require 
[5, 5, 5, 5, 10, 12, 33, 33]
k = 3
if __name__ == '__main__':
    print(max_kerner(num_list, k)) # pass 