import cv2
import os
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()

def getImageId():
  paths = [os.path.join('images', f) for f in os.listdir('images')]

  faces = []
  ids = []
  
  for path_img  in paths:
    img_face = cv2.cvtColor(cv2.imread(path_img), cv2.COLOR_BGR2GRAY)
    id = int(os.path.split(path_img)[-1].split('.')[1])
    ids.append(id)
    faces.append(img_face)
    
  return np.array(ids), faces

ids, faces = getImageId()

print("training...\n")

eigenface.train(faces, ids)
eigenface.write('declarantEigen.yml')

fisherface.train(faces, ids)
fisherface.write('declarantFisher.yml')

lbph.train(faces, ids)
lbph.write('declarantLBPH.yml')

print("\nsuccess!")