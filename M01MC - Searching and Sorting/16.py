def remove_duplicate(lst):
    dic = {}
    for i in lst:
        if i in dic:
            dic[i] += 1
        else:
            dic[i] = 1
    return list(dic.keys())

lst = [9, 9 , 8, 1, 1]
print(remove_duplicate(lst))

# (.venv) main14@main14-Aspire-A715-42G:
# ~/Downloads/AI$ /home/main14/Downloads/AI/.venv/bin/python 
# "/home/main14/Downloads/AI/M01MC - Searching and Sorting/16.py"
# [9, 8, 1]