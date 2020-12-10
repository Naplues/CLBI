def square_numbers(n):
    x = 1
    while x <= n:
        print(x * x, 'in square_numbers')
        yield x * x
        x += 1


for number in square_numbers(10):
    print(number)
