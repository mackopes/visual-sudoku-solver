from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout
from tensorflow.keras import Model


class DigitClassificationModel(Model):
    def __init__(self):
        super(DigitClassificationModel, self).__init__()
        self.dropout1 = Dropout(0.3)
        self.conv1 = Conv2D(64, 7, activation='relu')
        self.dropout2 = Dropout(0.3)
        self.conv2 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, x):
        x = self.dropout1(x)
        x = self.conv1(x)
        x = self.dropout2(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)
