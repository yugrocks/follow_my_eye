import win32api
import cv2
from detect import detector

def get_cursor_position():
    x, y = win32api.GetCursorPos()
    return x, y


def generate_data():
    cursor = []
    i = 20600
    cv2.namedWindow("preview")
    dtr = detector()
    vc = cv2.VideoCapture(0)
    
    if vc.isOpened(): #get the first frame
        rval, frame = vc.read()
        
    else:
        rval = False
        
    while rval:
        frame=cv2.flip(frame,1)
        eyes = dtr.detect(frame)
#        if len(eyes) > 0:
#            x,y,w,h = eyes[0]
#            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
#            eye = frame[y:y+h, x:x+w]
#            eye = cv2.cvtColor( eye, cv2.COLOR_RGB2GRAY )
#            eye = cv2.resize(eye, (50, 50))
#            cv2.imwrite('.\\Data\\eye_images\\eye_{}.jpg'.format(i), eye)
#            cursor.append(get_cursor_position())
#            i += 1
        for (x,y,w,h) in eyes:
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            eye = frame[y:y+h, x:x+w]
            eye = cv2.cvtColor( eye, cv2.COLOR_RGB2GRAY )
            eye = cv2.resize(eye, (50, 50))
            cv2.imwrite('.\\Data\\eye_images\\eye_{}.jpg'.format(i), eye)
            cursor.append(get_cursor_position())
            i += 1
            print(i)
            break
            
        cv2.imshow('preview',frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
    cv2.destroyWindow("preview")
    vc=None
    return cursor


cursor = generate_data()

