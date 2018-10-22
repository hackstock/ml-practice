import cv2
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(y_test[:3])
X_train = X_train.reshape((60000, 28 * 28))
X_train = X_train.astype('float32')
X_train /= 255

input_shape = (X_train.shape[1],)

X_test = X_test.reshape((10000, 28 * 28))
X_test = X_test.astype('float32')
X_test /= 255

y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=input_shape))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
callback = EarlyStopping(patience=2)
hist_callback = model.fit(X_train, y_train, validation_split=0.3, epochs=30, callbacks=[callback])

img = np.ones((600,600), dtype="uint8") * 255
img[100:500,100:500] = 0
pt1 = None
pt2 = None
is_drawing = False

def line(img,p1,p2):
    cv2.line(img,p1,p2,255,15)

def draw(event,x,y,flags,params):
    global pt1
    global pt2
    global img
    global is_drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        if is_drawing:
            pt1 = (x,y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing:
            pt2 = (x,y)
            line(img,pt1,pt2)
            pt1 = pt2
    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False

cv2.namedWindow("main")
cv2.setMouseCallback("main", draw)

while(True):
    cv2.imshow("main", img)
    key = cv2.waitKey(1) & 0xFF 
    if key == ord('q'):
        break
    elif key == ord('s'):
        is_drawing = True
    elif key == ord('c'):
        img[100:500,100:500] = 0
    elif key == ord('p'):
        char_image = img[100:500,100:500]
        char_image = cv2.resize(char_image,(28,28))
        input = char_image.reshape((28 * 28,))
        input = input.astype('float32')
        input /= 255
        print(input.shape)
        # result = model.predict_classes(np.array([X_test[0]]))
        result = model.predict_classes(np.array([input]))
        print("PREDICTION : ",result)

cv2.destroyAllWindows()