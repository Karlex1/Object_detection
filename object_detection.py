import cv2
import mediapipe as mp

# mp_hands=mp.solutions.hands
# mp_drawing=mp.solutions.drawing_utils
# hands=mp_hands.Hands(max_num_hands=2)

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

modelpath='ssd_mobilenet_v2.tflite'

base_options=python.BaseOptions(model_asset_path=modelpath)
options=vision.ObjectDetectorOptions(base_options=base_options,score_threshold=0.5)
detector=vision.ObjectDetector.create_from_options(options)

cap=cv2.VideoCapture(0)#0 for the first camera of device

while cap.isOpened():
    ret,frame=cap.read()
    if not ret:
        break
    image=mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    # srgb-standard Red green Blue and data.... is we know cv2 understandable format to mediapipe and mp.Image- convert numpy array to image format
    detection_result=detector.detect(image)
    
    for detection in detection_result.detections:
        bbox=detection.bounding_box
        start=(int(bbox.origin_x),int(bbox.origin_y))
        end=(int(bbox.origin_x+bbox.width),int(bbox.origin_y+bbox.height))
        category=detection.categories[0].category_name
        score=round(detection.categories[0].score,2)
        
        cv2.rectangle(frame,start,end,(0,248,2),2)
        cv2.putText(frame,f"{category}({score})",(start[0],start[1]-10),cv2.FONT_HERSHEY_DUPLEX,0.6,(255,2,0),2)
    cv2.imshow("MediaPipe Object Detector",frame)
    if cv2.waitKey(1)& 0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()