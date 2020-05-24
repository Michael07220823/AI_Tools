import cv2
import dlib
import numpy as np
from imutils import face_utils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Computer Vision\Face\models\shape_predictor_68_face_landmarks.dat")

img = cv2.imread("Computer Vision\Face\detection\many_people.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = detector(gray_img, 1)

for face in faces:
    shape = predictor(gray_img, face)
    shape = face_utils.shape_to_np(shape)

    (x, y, w, h) = face_utils.rect_to_bb(face)
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)

cv2.imshow("Face", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Source: https://hardliver.blogspot.com/2017/07/dlib-dlib.html