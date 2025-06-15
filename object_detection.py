import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image,ImageTk


from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class object_detection:
    def __init__(self,window):
        self.window=window
        self.window.title("Object detector")
        modelpath='ssd_mobilenet_v2.tflite'

        base_options=python.BaseOptions(model_asset_path=modelpath)
        options=vision.ObjectDetectorOptions(base_options=base_options,score_threshold=0.5)
        self.detector=vision.ObjectDetector.create_from_options(options)
        self.cap=cv2.VideoCapture(0)
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()
        self.update_frame()
        self.window.protocol("WW_DELETE_WINDOW",self.on_close)
    def update_frame(self):
        ret,frame=self.cap.read()
        if ret:
            mp_image=mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            # srgb-standard Red green Blue and data.... is we know cv2 understandable format to mediapipe and mp.Image- convert numpy array to image format
            
            detection_result=self.detector.detect(mp_image)
            
            for detection in detection_result.detections:
                bbox=detection.bounding_box
                start=(int(bbox.origin_x),int(bbox.origin_y))
                end=(int(bbox.origin_x+bbox.width),int(bbox.origin_y+bbox.height))
                category=detection.categories[0].category_name
                score=round(detection.categories[0].score,2)
                cv2.rectangle(frame,start,end,(0,248,2),2)
                cv2.putText(frame,f"{category}({score})",(start[0],start[1]-10),cv2.FONT_HERSHEY_DUPLEX,0.6,(255,2,0),2)
                
            img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            img=Image.fromarray(img)
            imgtk=ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0,0,anchor=tk.NW,image=imgtk)
            self.canvas.imgtk=imgtk
        self.window.after(10,self.update_frame)
    def on_close(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.window.destroy()
if __name__=="__main__":
    root = tk.Tk()
    app = object_detection(root)
    root.mainloop()
