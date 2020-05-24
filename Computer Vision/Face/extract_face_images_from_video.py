# Usage
# python extract_face_images_from_video.py -i testing-data/Trump.mp4 -o output -fm "F:/Master/Github/AI_Tools/Computer Vision/Face/models/face_detection/opencv_face_detector_uint8.pb" -fc "F:/Master/Github/AI_Tools/Computer Vision/Face/models/face_detection/opencv_face_detector.pbtxt" -f 1
# python extract_face_images_from_video.py -i 0 -o output -fm "F:/Master/Github/AI_Tools/Computer Vision/Face/models/face_detection/opencv_face_detector_uint8.pb" -fc "F:/Master/Github/AI_Tools/Computer Vision/Face/models/face_detection/opencv_face_detector.pbtxt" -f 1

import os
import cv2
import sys
import time
import argparse

    
def detectFaceOpenCVDnn(network=None, frame=None, conf_threshold = 0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)

    network.setInput(blob)
    detections = network.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x = int(detections[0, 0, i, 3] * frameWidth)
            y = int(detections[0, 0, i, 4] * frameHeight)
            w = int(detections[0, 0, i, 5] * frameWidth)
            h = int(detections[0, 0, i, 6] * frameHeight)
            bboxes = [x, y, w, h]
            cv2.rectangle(frameOpencvDnn, (x, y), (w, h), (0, 255, 0), int(round(frameHeight/150)), 8)

            # Save image
            cv2.imwrite(args["output"] + "\\{}.jpg".format(str(frame_count).zfill(5)), frame[y:h, x:w])
        else:
            bboxes = None
    return frameOpencvDnn, bboxes


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="Input video path or webcam index like 0.")
ap.add_argument("-o", "--output", required=True, help="Output images path.")
ap.add_argument("-fm", "--face_model", required=False, help="Specified face detection model file path.")
ap.add_argument("-fc", "--face_config", required=False, help="Specified face config file path.")
ap.add_argument("-df", "--DNN_framework", required=False, default="tensorflow", help="Specified DNN framework. tensorflow or caffe")
ap.add_argument("-f", "--flip", required=False, default=0, help="Flip frame.")
args = vars(ap.parse_args())

# OpenCV DNN supports 2 networks.
# 1. FP16 version of the original caffe implementation ( 5.4 MB )
# 2. 8 bit Quantized version using Tensorflow ( 2.7 MB )
# detect_face()使用，判斷使用哪一個深度學習框架，有Tensorflow和Caffe兩種
assert os.path.exists(args["face_config"]), "face_config path has problem !"
assert os.path.exists(args["face_model"]), "face_model path has problem !"
if args["DNN_framework"] == "caffe":
    network = cv2.dnn.readNetFromCaffe(args["face_config"], args["face_model"])
else:
    network = cv2.dnn.readNetFromTensorflow(args["face_model"], args["face_config"])


if not args["input"].isdigit():
    cap = cv2.VideoCapture(args["input"])
else:
    cap = cv2.VideoCapture(int(args["input"]))

hasFrame, frame = cap.read()

frame_count = 0
tt_opencvDnn = 0

while cap.isOpened():
    hasFrame, frame = cap.read()

    # Get video information
    video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    frame = cv2.flip(cv2.resize(frame, (800, 450)), int(args["flip"]))

    if not hasFrame:
        break
    frame_count += 1

    t = time.time()
    outOpencvDnn, bboxes = detectFaceOpenCVDnn(network, frame)

    tt_opencvDnn += time.time() - t
    fpsOpencvDnn = frame_count / tt_opencvDnn
    label = "OpenCV DNN ; FPS : {:.2f}".format(fpsOpencvDnn)
    cv2.putText(outOpencvDnn, label, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Face Detection Comparison", outOpencvDnn)

    if frame_count == 1:
        tt_opencvDnn = 0

    k = cv2.waitKey(10) & 0xFF
    if k == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        break