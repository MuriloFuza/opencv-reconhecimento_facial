import cv2 
import numpy as np


declarant = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)

sample = 1
number_samples = 12
id = input('Digit your name: ')
width, height = 480, 480
print("capturing the faces...")

while(True):
  connect_cam, img = cam.read()
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces_detected = declarant.detectMultiScale(img_gray, scaleFactor=1.5, minSize=(280, 280))
  
  for(x,y,w, h) in faces_detected:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    region = img[y:y + w, y:y + w]
    
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