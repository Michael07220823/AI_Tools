# Necessary package
# pip install mtcnn tensorflow-gpu >=2.0.0

import os
import cv2
import math
import dlib
from Timer import Timer
from mtcnn import MTCNN
from imutils import resize
from imutils.face_utils import rect_to_bb
from imutils.face_utils import FaceAligner


class Face_Alignment():
    @classmethod
    def eyes_center_coordinate(cls, x1,y1 ,x2,y2):
        '''計算出兩眼中心的座標'''
        (x, y) = (int((x1+x2) /2), int((y1+y2)/2))
        return (x, y)


    @classmethod
    def angle_2points(cls, x1, y1, x2, y2):
        '''使用兩眼的斜率，計算出人臉須要旋轉的角度'''
        rotate_angle = math.atan2(y2-y1, x2-x1) * 180 / math.pi
        return rotate_angle


    @classmethod
    def area_expand(cls, x, y, w, h):
        '''擴增裁切的範圍，以避免裁切到人臉'''
        nx = int(x - w/8)
        nw = int(1.25 * w)
        ny = int(y - h/8)
        nh = int(1.3 * h)

        if nx < 0:
            nx = 0
        if ny < 0:
            ny = 0
        return (nx, ny, nw, nh)


    @classmethod
    def mtcnn_alignment(cls, image, save_path=None, show_image=False):
        detector = MTCNN()

        try:
            if type(image) == str:
                # mtcnn only receive rgb or bgr format, image need to has three channels.
                image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)

            # Detect face and get face five point.
            result = detector.detect_faces(image)

            if len(result) > 0:
                # Face five point: left_eye, right_eye, nose, left_mouth, right_mouth.
                face_point = result[0].get('keypoints')
                
                # ROI
                x, y, w, h = result[0]['box']

                # Get image height and width.
                (img_h, img_w) = image.shape[:2]

                # Face point.
                x1, y1 = face_point['left_eye']
                x2, y2 = face_point['right_eye']
                
                # Compute rotate angle.
                rotate_angle = cls.angle_2points(x1, y1, x2, y2)

                # Rotete image.
                M = cv2.getRotationMatrix2D(face_point["nose"], rotate_angle, scale=0.9)
                orig_rotated = cv2.warpAffine(image, M, (img_w, img_h))
                
                nx, ny, nw, nh = cls.area_expand(x, y, w, h)
                face_align = cv2.cvtColor(orig_rotated[ny:ny+nh, nx:nx+nw], cv2.COLOR_RGB2BGR)

                # Show image.
                if show_image:
                    cv2.imshow("Image", resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), width=640))
                    cv2.imshow("Alignment face", resize(face_align, width=180))
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                if save_path:
                    cv2.imwrite(save_path, face_align, [cv2.IMWRITE_JPEG_QUALITY, 100])
                return face_align
            else:
                print("[INFO] MTCNN not detect the face !")
        except cv2.error as e:
            print("[ERROR] %s" % e)


    @classmethod
    def dlib_5point_alignment(cls, image, shape_5_landmark_file, save_path=None, show_image=False):

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(shape_5_landmark_file)
        fa = FaceAligner(predictor, desiredFaceWidth=256)

        if type(image) == str:
            # load the input image, resize it, and convert it to grayscale
            image = cv2.imread(image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # show the original input image and detect faces in the grayscale
        # image
        rects = detector(gray, 2)

        if len(rects) > 0:
            for rect in rects:
                # extract the ROI of the *original* face, then align the face
                # using facial landmarks
                (x, y, w, h) = rect_to_bb(rect)
                face_align = fa.align(image, gray, rect)
            
            # Show image.
            if show_image:
                cv2.imshow("Face Alignment face", face_align)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if save_path:
                cv2.imwrite(save_path, face_align, [cv2.IMWRITE_JPEG_QUALITY, 100])
            
            return face_align
        else:
            print("[INFO] Dlib_5 not detect the face !")


    @classmethod
    def dlib_68point_alignment(cls, image, shape_68_landmark_file, save_path=None, show_image=False):
    
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(shape_68_landmark_file)
        fa = FaceAligner(predictor, desiredFaceWidth=256)

        if type(image) == str:
            # load the input image, resize it, and convert it to grayscale
            image = cv2.imread(image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # show the original input image and detect faces in the grayscale
        # image
        rects = detector(gray, 2)

        if len(rects) > 0:
            for rect in rects:
                # extract the ROI of the *original* face, then align the face
                # using facial landmarks
                (x, y, w, h) = rect_to_bb(rect)
                face_align = fa.align(image, gray, rect)
            
            # Show image.
            if show_image:
                cv2.imshow("Face Alignment face", face_align)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            if save_path:
                cv2.imwrite(save_path, face_align, [cv2.IMWRITE_JPEG_QUALITY, 100])
            return face_align
        else:
            print("[INFO] Dlib_68 not detect the face !")


if __name__ == "__main__":
    image = "C:/Users/overcomer/Pictures/Camera Roll/left.jpg"
    # image = "C:/Users/overcomer/Pictures/Camera Roll/right2.jpg"
    # image = "data\\test\\white_board.jpg"
    face_aligner = Face_Alignment()
    with Timer("MTCNN"):
        mtcnn = face_aligner.mtcnn_alignment(image, save_path="mtcnn.jpg", show_image=False)
    with Timer("Dlib_5"):
        dlib_5 = face_aligner.dlib_5point_alignment(image, "models\\landmark\\shape_predictor_5_face_landmarks.dat", save_path="dlib_5.jpg", show_image=False)
    with Timer("Dlib_68"):
        dlib_68 = face_aligner.dlib_68point_alignment(image, "models\\landmark\\shape_predictor_68_face_landmarks.dat", save_path="dlib_68.jpg", show_image=False)