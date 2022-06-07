import cv2
import numpy as np
from elements.yolo import OBJ_DETECTION
from upload_thru_iot import Upload_thru_iot
import sys
import os,datetime

class JetsonYolo: 

    def __init__(self, connection_string,image_path="/yolo_image_recognition"):

        if(not os.path.exists(image_path)):
            os.makedirs(image_path)
        self.image_path=image_path    
        self.Object_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                        'hair drier', 'toothbrush' ]

        self.Object_colors = list(np.random.rand(80,3)*255)
        self.Object_detector = OBJ_DETECTION('weights/yolov5s.pt', self.Object_classes)
        self.connection_string=connection_string
        print("all variables loaded")

    def __gstreamer_pipeline(self,
        capture_width=1024,
        capture_height=768,
        display_width=1024,
        display_height=768,
        framerate=30,
        flip_method=0,
    ):
        print("preparing GSTREAMER pipeline")
        return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
        )

    def detect_and_upload(self):

        cap = cv2.VideoCapture(self.__gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
        image_uploader=Upload_thru_iot(self.connection_string)
        #cap=cv2.VideoCapture('/dev/video0')
        if cap.isOpened():
            try:
                while True:
                    #window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
                    # # Window
                    #print("camera is available")
                    ret, frame = cap.read()
                    if ret:
                        # detection process
                        #print("frame is available")
                        objs = self.Object_detector.detect(frame)
                        # plotting
                        upload_image=False
                        for obj in objs:
                            # print(obj)
                            label = obj['label']
                            score = obj['score']
                                
                            if(score>.70):
                                print("objects with greater than 70 percent confidence are detected")
                                upload_image=True
                                [(xmin,ymin),(xmax,ymax)] = obj['bbox']
                                color = self.Object_colors[self.Object_classes.index(label)]
                                frame = cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2) 
                                frame = cv2.putText(frame, f'{label} ({str(score )})', (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX , 0.75, color, 1, cv2.LINE_AA)
                        if upload_image:
                            print("uploading image to azure")
                            this_moment=datetime.datetime.now()
                            filename=("received_image_{year:x}{month:x}{day:x}{hour:x}{minute:x}{sec:x}.jpg".format(year=this_moment.year,month=this_moment.month,day=this_moment.day,hour=this_moment.hour,minute=this_moment.minute,sec=this_moment.second))
                            filepath=os.path.join(self.image_path,filename)
                            cv2.imwrite(filepath,frame)
                            image_uploader.upload_image(filepath)
            except KeyboardInterrupt:
                cap.release()
                cv2.destroyAllWindows()
            except:
                print("error occurred")
                cap.release()
                cv2.destroyAllWindows()                 
        else:
            print("unable to read camera")                

           

def main(args):
    print(args[1])
    thisObj=JetsonYolo(args[1])
    thisObj.detect_and_upload()
    

if __name__ == '__main__':
    main(sys.argv)
