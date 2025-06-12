
def count_word(file_path) :
    counter = {}
    with open(file_path, 'r') as file:
        content = file.read()
        for word in content.replace(',', '').replace('.', '').split():
            if word in counter:
                counter[word] +=1
            else:
                counter[word] = 1
    return counter

if __name__ == '__main__':
    path = '/home/main14/Downloads/AI/M01MC - Searching and Sorting/P1_data.txt'
    dic = count_word(path)
    assert dic['who'] == 3
    print(dic['man']) # = 6
    