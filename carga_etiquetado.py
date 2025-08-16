import cv2
import os
import numpy as np

from captura_rostros import dataPath

dataPath = './Data'
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)
labels = []
facesData = []
label = 0
for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('Leyendo las imágenes')

    for fileName in os.listdir(personPath):
        print('Rostros: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath + '/' + fileName, 0))
    label = label + 1

# Métodos para entrenar el reconocedor utilizando EigenFace, FisherFace y LBPH
face_recognizer_Eigen = cv2.face.EigenFaceRecognizer.create()
face_recognizer_Fisher = cv2.face.FisherFaceRecognizer.create()
face_recognizer_LBPHF = cv2.face.LBPHFaceRecognizer.create()

# Entrenando el reconocedor de rostros
print("Entrenando...")
face_recognizer_Eigen.train(facesData, np.array(labels))
face_recognizer_Fisher.train(facesData, np.array(labels))
face_recognizer_LBPHF.train(facesData, np.array(labels))

# Almacenando el modelo obtenido
face_recognizer_Eigen.write('eigenFace_model.yml')
face_recognizer_Fisher.write('fisherFace_model.yml')
face_recognizer_LBPHF.write('LBPHFace_model.yml')
print("Modelos almacenados...")