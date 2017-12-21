import cv2
import numpy as np


class detector:
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(r'cascades\\haarcascade_eye.xml')
        
    def detect(self, img):
        # take a grayscale image and return rectangles of eyes
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes = self.face_cascade.detectMultiScale(self.gray, 1.3, 5)
        return eyes
