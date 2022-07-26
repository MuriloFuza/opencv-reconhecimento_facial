import cv2 
import numpy as np


declarant = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
declarant_eye = cv2.CascadeClassifier("haarcascade_eye.xml")
cam = cv2.VideoCapture(0)

sample = 1
number_samples = 25
id = input('Digit your name: ')
width, height = 220, 220
print("capturing the faces...")

while(True):
  connect_cam, img = cam.read()
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces_detected = declarant.detectMultiScale(img_gray, scaleFactor=1.5, minSize=(150, 150))
  
  for(x,y,w, h) in faces_detected:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    region = img[y:y + w, y:y + w]
    region_gray_eye = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    eye_detected = declarant_eye.detectMultiScale(region_gray_eye)
    
    for(ex, ey, ew, eh) in eye_detected:
      cv2.rectangle(region, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
      
      if cv2.waitKey(1) & 0xFF == ord('q'):
        if np.average(img_gray) > 110:
          img_face = cv2.resize(img_gray[y:y + h, x:x + w], (width, height))
          cv2.imwrite("images/pessoa."+ str(id) + "." + str(sample)+".jpg", img_face)
          print("captured" + str(sample) + "success!")
          sample += 1

    

  
  cv2.imshow("Face", img)
  cv2.waitKey(1)
  
  if(sample >= number_samples + 1):
    break
  
print("finish!")
cam.release()
cv2.destroyAllWindows()  