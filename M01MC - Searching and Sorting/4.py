import numpy as np

def levenshtein_distance(token1, token2):
    # matrix
    n1 = len(token1)
    n2 = len(token2)
    matrix = np.zeros((n1 + 1, n2 + 1))
    # comparation with empty string is the basecase
    for i in range(1, n1+1):
        for j in range(1, n2+1):
            if i == 1 or j == 1:
                matrix[i][j] = i + j - 1
            else:
                ins = matrix[i-1][j] + 1
                dele = matrix[i][j-1] + 1 
                sub = matrix[i-1][j-1] + (token1[i-1] != token2[j-1])
                matrix[i][j] = min(ins, dele, sub)
    return matrix[n1-1][n2-1]

print(levenshtein_distance("hi", "hello"))
assert levenshtein_distance("hi", "hello") == 4.0
print(levenshtein_distance("hola", "hello"))

# (.venv) main14@main14-Aspire-A715-42G:
# ~/Downloads/AI$ /home/main14/Downloads/AI/.venv/bin/python "/home/main14/Downloads/AI/M01MC - Searching and Sorting/4.py"
# 4.0
# 3.0