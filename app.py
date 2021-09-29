import numpy as np
import tensorflow as tf
import cv2 as cv
from tensorflow import keras
from keras.models import model_from_json
from tensorflow.python.ops.functional_ops import While

################### Load Model#################
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
new_model = model_from_json(loaded_model_json)
# load weights into new model
new_model.load_weights("best_model.hdf5")
print("Loaded model from disk")

################## Model Summary #####################

new_model.summary()

################## Compile ######################
new_model.compile(optimizer = "adam",loss = "categorical_crossentropy",metrics=["accuracy"])
#################### Parameter #####################

width = 600
height = 400
threshold = 0.5
####################### Web Cam ######################
cap = cv.VideoCapture(1)
cap.set(3,width)
cap.set(4,height)

def preprocessing(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.equalizeHist(img)
    img = img/255
    return img


while True:
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv.resize(img,(150,150))
    img = preprocessing(img)
    cv.imshow("Processed Image", img)
    img = img.reshape(1,150,150,1)
    #predict
    classIndex = int(new_model.predict_classes(img))
    prediction = new_model.predict(img)
    probval = np.amax(prediction)
    print(classIndex, probval)

    if probval> threshold:
        cv.putText(imgOriginal, str(classIndex)+ " " + str(probval),
                   (50,50),cv.FONT_HERSHEY_COMPLEX,
                   1,(0,0,255), 1)



    cv.imshow("Original Image", imgOriginal)



    if cv.waitKey(1) & 0xFF == ord('q'):
        break




