import os
import cv2
import dlib

class Face_Detect():
    def haar_detect(self, haar_file=str(), image=None, show_image=False):
        roi = None
        face = None

        faceCascade = cv2.CascadeClassifier(haar_file)
        
        # 判斷是否要讀取圖片
        if type(image) == str:
            image = cv2.resize(cv2.imread(image), (640, 480))
        copy_image = image.copy()

        gray = cv2.cvtColor(copy_image, cv2.COLOR_BGR2GRAY)
    
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=6,
            minSize=(30, 30),
        )
    
        # Draw a rectangle around the faces
        for roi in faces:
            (x, y, w, h) = roi
            cv2.rectangle(copy_image, (x, y), (x+w, y+h), (255, 155, 0), 2)
            face = image[y:y+h, x:x+w]

            if show_image:
                cv2.imshow('Haar Face', face)
                cv2.imshow('Haar Original', copy_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return roi, image, face

    
    def dlib_detect(self, image=None, show_image=False):
        roi = None
        face =None
        
        detector = dlib.get_frontal_face_detector()
        
        # 判斷是否要讀取圖片
        if type(image) == str:
            image = cv2.resize(cv2.imread(image), (640, 480))
        copy_image = image.copy()
        faces = detector(copy_image, 1)

        for face in faces:
            x = face.left()
            y = face.top()
            w = face.right()
            h = face.bottom()
            # cv2.circle(copy_image, (int((x+w)/2), int((y+h)/2)), 100, (0, 155, 255), 2)
            cv2.rectangle(copy_image, (x, y), (w, h), (255, 155, 0), 2)

            face = image[y:h, x:w]

            if show_image:
                cv2.imshow("Dlib Face", cv2.resize(face, (250, 250)))
                cv2.imshow("Dlib Original", copy_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return roi, image, face
        

    def dnn_detect(self, 
                   face_config=str(),
                   face_model=str(), 
                   dnn_framework="tensorflow",
                   image=None, 
                   conf_threshold=0.7, 
                   show_image=False, 
                   ):

        assert os.path.exists(face_config), "face_config path has problem !"
        assert os.path.exists(face_model), "face_model path has problem !"
        if dnn_framework == "caffe":
            network = cv2.dnn.readNetFromCaffe(face_config, face_model)
        else:
            network = cv2.dnn.readNetFromTensorflow(face_model, face_config)

        # 判斷是否要讀取圖片
        if type(image) == str:
            image = cv2.resize(cv2.imread(image), (640, 480))

        # init roi and face
        roi = None
        face = None

        copy_image = image.copy()
        frameHeight = copy_image.shape[0]
        frameWidth = copy_image.shape[1]
        # This image net to three chaneel image format.
        blob = cv2.dnn.blobFromImage(copy_image, 1.0, (300, 300), [104, 117, 123], False, False)

        network.setInput(blob)
        detections = network.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x = int(detections[0, 0, i, 3] * frameWidth)
                y = int(detections[0, 0, i, 4] * frameHeight)
                w = int(detections[0, 0, i, 5] * frameWidth)
                h = int(detections[0, 0, i, 6] * frameHeight)
                # face coordiante
                if x or y or w or h:
                    roi = [x, y, w, h]
                    face = image[y:h, x:w]
                    cv2.rectangle(copy_image, (x, y), (w, h), (255, 155, 0), 2)
                else:
                    continue
        if type(face) != type(None):
            if show_image:
                cv2.imshow("DNN face", cv2.resize(face, (250, 250)))
                cv2.imshow("DNN Original", copy_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        return roi, image, face


if __name__ == "__main__":
    detector = Face_Detect()
    roi, image, face = detector.dnn_detect(face_config="F:/Master/Github/LBPH_Face_Recognition/models/face_detection/dnn/opencv_face_detector.pbtxt",
                                           face_model="F:/Master/Github/LBPH_Face_Recognition/models/face_detection/dnn/opencv_face_detector_uint8.pb",
                                           image="d:/2.jpg",
                                           show_image=True
                                          )
    
    roi, image, face = detector.dlib_detect(image="d:/2.jpg", show_image=True)

    roi, image, face = detector.haar_detect(haar_file="F:/Master/Github/LBPH_Face_Recognition/models/face_detection/haarcascade_frontalface_default.xml",
                                            image="d:/2.jpg",
                                            show_image=True)

