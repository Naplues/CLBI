# -*- coding: utf-8 -*-


class Animal:
    def __init__(self, name):
        self.name = name

    def show(self):
        print(self.name)


class Dog(Animal):
    def __init__(self, name, age=12):
        super().__init__(name)
        self.age = age

    def show(self):
        print('Dog', self.name, self.age)


class Cat(Animal):
    def __init__(self, name, age=12):
        super().__init__(name)
        self.age = age

    def show(self):
        print('Cat', self.name, self.age)


def fun(animal: Animal):
    animal.show()


if __name__ == '__main__':
    fun(Dog('aaa'))
    fun(Cat('bbb'))
