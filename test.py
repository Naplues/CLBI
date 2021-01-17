



from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, units=784),  # 代码感觉有误，应该是input_shape=(784,)
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

if __name__ == '__main__':
    print(model)
