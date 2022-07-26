import cv2

detector_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.EigenFaceRecognizer_create()
recognizer.read("declarantEigen.yml")

width, height = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

cam = cv2.VideoCapture(0)

while(True):
  connect_cam, img = cam.read()
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces_detected = detector_face.detectMultiScale(img_gray, scaleFactor=1.5, minSize=(150, 150))
  
  for(x,y,w, h) in faces_detected:
    img_face = cv2.resize(img_gray[y:y + w, y:y + w], (width, height))
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    id, confidence = recognizer.predict(img_face)
    if id == 1:
      nome = 'Ivo'
    if id == 2:
      nome = 'Murilo'
    else:
      nome = 'Thiago'
    cv2.putText(img, nome, (x,y + (h + 30)), font, 3, (0,0,255))
  
  cv2.imshow("Face", img)
  cv2.waitKey(1)
  
cam.release()
cam.destroyAllWindowns()