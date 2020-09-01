from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D
from tensorflow.keras import Model


class DigitClassificationModel(Model):
    def __init__(self):
        super(DigitClassificationModel, self).__init__()
        self.conv1 = Conv2D(32, 5, strides=2, activation='relu')
        self.conv2 = Conv2D(64, 3, activation='relu')
        self.maxpool1 = MaxPool2D(pool_size=2)
        self.flatten = Flatten()
        self.dropout1 = Dropout(0.25)
        self.d1 = Dense(128, activation='relu')
        self.dropout2 = Dropout(0.5)
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.d1(x)
        x = self.dropout2(x)
        return self.d2(x)
