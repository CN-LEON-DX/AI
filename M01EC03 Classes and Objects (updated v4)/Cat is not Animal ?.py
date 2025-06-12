class Animal:
    def __init__(self, name):
        self.name = name
        self.__age = 10 #private

    def make_sound(self):
        return "Some generic animal sound"

class Cat(Animal):
    def __init__(self, name, bread):
        super().__init__(name)
        self.bread = bread
    def info(self):
        return f"{self.name} is a CAT of bread {self.bread}"

my_cat = Cat("Fuzzyfox", bread="Siamese")
print(my_cat.info())
print(my_cat.make_sound())

my_cat.__age = 10
print(my_cat.__age)