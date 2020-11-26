class Bird:
    eyes = 'two'

    def __init__(self, color, feet):
        self.color = color
        self.feet = feet

    def call(self, cd):
        print('This bird:', cd)

    @classmethod
    def fly(cls, p):
        print('class method', p)

    @staticmethod
    def f(p):
        print('static method', p)


a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ]
b = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9,)
print(a[0::2])
print(b[1::2])
c = set('abcddsa')
d = set('sfibs')

print(c, d)
print(c - d)
print(c | d)
print(c & d)
num = str((1202 ** 569) - (1012 ** 569))[::121]
print((num))