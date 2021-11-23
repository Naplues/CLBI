# -*- coding: utf-8 -*-

import sys


class A:
    def __init__(self, name='ABC', age=20):
        self.name = name
        self.age = age

    def show(self):
        print(self.name, self.age)


class B(A):
    def __init__(self, name, age):
        super().__init__(name, age)

    pass


if __name__ == '__main__':
    #
    b = B('gzq', 12)
    b.show()
    c = B()
    c.show()
    pass
