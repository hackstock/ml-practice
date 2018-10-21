import cv2
import numpy as np
from keras.models import model_from_json

file = open("model.json", "r")
model_def = file.read()
file.close()
model = model_from_json(model_def)
model.load_weights("model.h5")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

img = np.zeros((600,600,3), dtype="uint8")
img[100:500,100:500] = (255,255,255)
pt1 = None
pt2 = None
is_drawing = False

def line(img,p1,p2):
    cv2.line(img,p1,p2,(255,0,0),10)

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
        img[100:500,100:500] = (255,255,255)
    elif key == ord('p'):
        char_image = img[100:500,100:500]
        char_image = cv2.cvtColor(char_image, cv2.COLOR_BGR2GRAY)
        char_image = cv2.resize(char_image,(28,28))
        input = char_image.reshape((28 * 28,))
        input = input.astype('float32')
        input /= 255
        print(input.shape)
        result = model.predict_proba(np.array([input]))
        print("PREDICTION : ",result)

cv2.destroyAllWindows()