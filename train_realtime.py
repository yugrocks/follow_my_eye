import cv2
import numpy as np
from detect import detector
from keras.models import model_from_json
import win32api


def move(x, y):
    win32api.SetCursorPos((x , y))

def locate_cursor():
    return win32api.GetCursorPos()

def load_model():
    try:
        json_file = open('model2/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("model2/weights.hdf5")
        print("Model successfully loaded from disk.")
        #compile again
        model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mse'])
        return model
    except:
        print("""Model not found. Please train the CNN by running the script 
               cnn_train.py. Note that the training and test samples should be properly 
                 set up in the dataset directory.""")
        return None
    
def realtime():
    #initialize preview
    cv2.namedWindow("preview")
    dtr = detector()
    vc = cv2.VideoCapture(0)
    model = load_model()
    max_X = 20 # max movement of x possible
    max_Y = 20 # max movement of y possible
    prev_X, prev_Y = locate_cursor() # initial position
    if vc.isOpened(): #get the first frame
        rval, frame = vc.read()
    else:
        rval = False
    while rval:
        frame=cv2.flip(frame,1)
        eyes = dtr.detect(frame)
        for (x,y,w,h) in eyes:
            img = cv2.cvtColor( frame[y:y+h, x:x+w], cv2.COLOR_RGB2GRAY )
            img = cv2.resize(img, (50, 50), interpolation = cv2.INTER_AREA)
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            img=img.reshape((1,)+img.shape+(1,))/255.
            
            y_pred = model.predict(img) #predict its class
            x = int(y_pred[0][0])
            y = int(y_pred[0][1])
            if x - prev_X > max_X:
                x = prev_X +  max_X
            elif x - prev_X < -max_X:
                x = prev_X - max_X
            if y - prev_Y > max_Y:
                y = prev_Y + max_Y
            elif y - prev_Y < -max_Y:
                y = prev_Y - max_Y
            move(x,y)
            prev_X = x; prev_Y = y
            break
        cv2.imshow('preview',frame)
        
        
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
    cv2.destroyWindow("preview")
    vc=None
    
realtime()
