import cv2
import math
import dlib
import time
import numpy
import imutils
from mtcnn import MTCNN
from imutils.face_utils import rect_to_bb
from imutils.face_utils import FaceAligner


class Face_Align():
    def __init__(self, image):
        self.image = image

    def eyes_distance(self, p1, p2):
        length = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)/2
        return int(length)


    def eyes_center_coordinate(self, p1, p2):
        (x, y) = (int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2))
        return (x, y)


    def angle_2points(self, p1, p2):
        r_angle = math.atan2(p1[1] - p2[1], p1[0] - p2[0])
        rotate_angle = abs(r_angle * 180 / math.pi)
        return rotate_angle


    def area_expand(self, roi):
        x ,y, w, h = roi
        nx = int(x - w/8)
        nw = int(1.25 * w)
        ny = int(y - h/8)
        nh = int(1.25 * h)

        if nx < 0:
            nx = 0
        if ny < 0:
            ny = 0
        return (nx, ny, nw, nh)


    def mtcnn_alignment(self, show_image=False):
        start = time.time()

        detector = MTCNN()

        if type(self.image) == str:
            # mtcnn only receive rgb or bgr format, image need to has three channels.
            image = cv2.cvtColor(cv2.imread(self.image), cv2.COLOR_BGR2RGB)
        else:
            image = self.image.copy()
        
        # Detect face and get face five point.
        result = detector.detect_faces(image)

        # Face five point: left_eye, right_eye, nose, left_mouth, right_mouth.
        face_point = result[0].get('keypoints')
        
        # ROI
        x, y, w, h = result[0]['box']

        # Get image height and width.
        (img_h, img_w) = image.shape[:2]

        # Face point.
        left_eye = face_point['left_eye']
        right_eye = face_point['right_eye']
        nose = face_point['nose']
        
        # Compute rotate angle.
        rotate_angle = self.angle_2points(face_point['left_eye'], face_point['right_eye'])

        # Rotete image.
        M = cv2.getRotationMatrix2D(face_point['nose'], 180-rotate_angle, scale=1.0)
        orig_rotated = cv2.warpAffine(image, M, (img_w, img_h))

        # Detect face again after rotated image.
        result = detector.detect_faces(orig_rotated)

        # New ROI
        x, y, w, h = result[0]['box']

        # RGB to BGR.
        face_align = cv2.cvtColor(orig_rotated[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
        print("Alignmented face Cost %.2f secs" % (time.time() - start))

        # Show image.
        if show_image:
            cv2.imshow("MTCNN", face_align)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return face_align


    def dlib_5point_alignment(self, shape_5_landmark_file, show_image=False):
        start = time.time()

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(shape_5_landmark_file)
        fa = FaceAligner(predictor, desiredFaceWidth=256)

        if type(self.image) == str:
            # load the input image, resize it, and convert it to grayscale
            image = cv2.imread(self.image)
        else:
            image = self.image.copy()

        # image = imutils.resize(image, width=1200)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # show the original input image and detect faces in the grayscale
        # image
        rects = detector(gray, 2)

        # loop over the face detections
        for rect in rects:
            # extract the ROI of the *original* face, then align the face
            # using facial landmarks
            (x, y, w, h) = rect_to_bb(rect)
            face_orig = imutils.resize(image[y:y + h, x:x + w], width=256)
            face_align = fa.align(image, gray, rect)

        print("Aligmented face cost %.2f secs." % (time.time() - start))
        
        # Show image.
        if show_image:
            cv2.imshow("MTCNN", face_align)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return face_align


    def dlib_68point_alignment(self, shape_68_landmark_file, show_image=False):
        start = time.time()

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(shape_68_landmark_file)
        fa = FaceAligner(predictor, desiredFaceWidth=256)

        if type(self.image) == str:
            # load the input image, resize it, and convert it to grayscale
            image = cv2.imread(self.image)
        else:
            image - self.image.copy()

        # image = imutils.resize(image, width=1200)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # show the original input image and detect faces in the grayscale
        # image
        rects = detector(gray, 2)

        # loop over the face detections
        for rect in rects:
            # extract the ROI of the *original* face, then align the face
            # using facial landmarks
            (x, y, w, h) = rect_to_bb(rect)
            face_orig = imutils.resize(image[y:y + h, x:x + w], width=256)
            face_align = fa.align(image, gray, rect)

        print("Aligmented face cost %.2f secs." % (time.time() - start))
        
        # Show image.
        if show_image:
            cv2.imshow("MTCNN", face_align)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return face_align


if __name__ == "__main__":
    import cv2

    face_aligner = Face_Align("testing-data/lady.jpg")
    mtcnn_align = face_aligner.mtcnn_alignment(show_image=True)
    dlib_5 = face_aligner.dlib_5point_alignment("models\shape_predictor_5_face_landmarks.dat", show_image=True)
    dlib_68 = face_aligner.dlib_68point_alignment("models\shape_predictor_68_face_landmarks.dat", show_image=True)