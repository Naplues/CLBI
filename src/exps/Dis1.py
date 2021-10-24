# -*- coding:utf-8 -*-

def extract():
    with open('a.csv', 'r') as file:
        d = file.readlines()

    text = ''
    flag = False

    last_effort = 0

    for line in d:
        if '=' in line:
            last_name = line.split(' ')[-2]
            text += last_name + '\n'
            print(last_name)
        else:
            if flag:
                text += line
                flag = False
            else:
                flag = True

    with open('r.csv', 'w') as file:
        file.write(text)


def trans():
    with open('r.csv', 'r') as file:
        data = file.readlines()

    text = ''
    last_name = data[0].strip()
    temp = ''
    for line in data[1:]:
        if '[' not in line:
            text += last_name + ',' + temp + '\n'
            last_name = line.strip()
            temp = ''
        else:
            temp += line.strip().replace('[', '').replace(']', '') + ', '

    text += last_name + ',' + temp + '\n'

    with open('r2.csv', 'w') as file:
        file.write(text)


if __name__ == '__main__':
    trans()
