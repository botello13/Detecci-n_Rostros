import cv2
import os

from carga_etiquetado import face_recognizer_Fisher

# Ruta donde se almacenan las imágenes de entrenamiento
dataPath = './Data'
# Lista de imágenes en el directorio Data
imagePaths = os.listdir(dataPath)
print('imagePaths=', imagePaths)

# Creando el reconocedor de rostros
face_recognizer_Eigen = cv2.face.EigenFaceRecognizer.create()
#face_recognizer_Fisher = cv2.face.FisherFaceRecognizer.create()
#face_recognizer_LBPHF = cv2.face.LBPHFaceRecognizer.create()

# Leyendo el modelo
face_recognizer_Eigen.read('./eigenFace_model.yml')
#face_recognizer_Fisher.read('./fisherFace_model.yml')
#face_recognizer_LBPHF.read('./LBPHFace_model.yml')

# Usar la webcam (índice 0)
cap = cv2.VideoCapture(0)
# Cargando el clasificador de rostros
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer_Eigen.predict(rostro)
        cv2.putText(frame, '{}'.format(result), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

        if result[1] < 5700:
            cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()