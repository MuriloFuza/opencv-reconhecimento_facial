from email.policy import strict
import cv2
from keras.utils import load_img
import numpy as np

detector_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create(1, 4, 4, 4,50)
recognizer.read("declarantLBPH.yml")

image = load_img("./thiago.JPG")

image = np.array(image)

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

id, confidence = recognizer.predict(image)

print(id, confidence)