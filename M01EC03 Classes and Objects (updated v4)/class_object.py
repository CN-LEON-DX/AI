# In Python, everything is an object.
class Square():
    def __init__(self, side):
        self.side = side
    def compute_area(self):
        return self.side * self.side
    def describe(self):
        print(f'Side is {self.side}')


if __name__ == '__main__':
    list_square = [Square(13), Square(21), Square(3), Square(24)]
    list_square.sort(key=lambda x: x.side)
    
    for square  in list_square:
        square.describe()
    

# (.venv) main14@main14-Aspire-A715-42G:
# ~/Downloads/AI$ /home/main14/Downloads/AI/.venv/bin/python 
# "/home/main14/Downloads/AI/M01EC03 Classes and Objects (updated v4)/class_object.py"
# Side is 3
# Side is 13
# Side is 21
# Side is 24