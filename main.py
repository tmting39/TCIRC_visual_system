import numpy as np
import cv2
import mediapipe as mp
import pyttsx3
import threading
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


cap = cv2.VideoCapture('test.mp4')
#cap = cv2.VideoCapture(0)


frame_size = cap.read()[1].shape
correction_distance = 140
correction_size = 170


object_size = {'person':165,
               'cars':160,
               'truck':320,
               'motorcycle':100}



MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 3
FONT_THICKNESS = 2
TEXT_COLOR = (255, 0, 0)  # red

engine = pyttsx3.init()
engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0')

t = 0
stop=0
txt=''
def sy():
    while(stop==0):
        global t
        if t >= 20:
            print(txt)
            engine.say(txt)
            engine.runAndWait()
            t = 0
        time.sleep(2)
    print('stop')
    return

temp = threading.Thread(target=sy)

def visualize(
    image,
    detection_result
) -> np.ndarray:
    for detection in detection_result.detections:

        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

        if not object_size.get(category_name)==None:
            object_distance = frame_size[1] * correction_distance * object_size[category_name] / (correction_size * bbox.height * 100)
            object_distance=round(object_distance,1)
            direction = 'front'
            if bbox.origin_x + bbox.width / 2 < frame_size[0] / 3:
                direction = 'front-left'
            elif bbox.origin_x + bbox.width / 2 > frame_size[0] * 2 / 3:
                direction = 'front-right'

            global txt
            txt = 'There is a ' + category_name + ' ' + str(object_distance) + ' meters in ' + direction


    return image



base_options = python.BaseOptions(model_asset_path='efficientdet_lite2.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5,)
detector = vision.ObjectDetector.create_from_options(options)



temp.start()

while True:
    ret, frame=cap.read()
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect(image)

    image_copy = np.copy(image.numpy_view())
    annotated_image = visualize(image_copy, detection_result)
    cv2.imshow("result",annotated_image)
    t+=1;
    if cv2.waitKey(1) == ord('q'):
        stop = 1
        cap.release()
        cv2.destroyAllWindows()
        temp.join()
        break
