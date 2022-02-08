import cv2, json
from keras import Model
from keras.models import load_model
from Stream import Stream
from lib.utils.utils import get_yolo_boxes
from lib.utils.bbox import draw_boxes
from numpy import ndarray
from threading import Thread, Lock


#This is for rtsp video streams ex. rtsp://<username>:<pwd>@192.168.1.xxx:xxx/cam1/onvif-h264
def startRTSP(input_path:str, config:json) -> None:

    infer_model = load_model(config['train']['saved_weights_name'], compile=False)
    net_h, net_w = 416, 416
    obj_thresh, nms_thresh = 0.95, 0.45

    video_reader = cv2.VideoCapture(input_path)
    video_reader.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # the main loop
    batch_size  = 1
    images      = []
    
    while True:
        ret_val, image = video_reader.read()

        image = cv2.resize(image, (960, 512))

        if ret_val == True: images += [image]

        if (len(images)==batch_size) or (ret_val==False and len(images)>0):
            batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)

            for i in range(len(images)):
                draw_boxes(images[i], batch_boxes[i], config['model']['labels'], obj_thresh) 
                cv2.imshow('video with bboxes', images[i])
            images = []
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows() 

class RSTPStream(Stream):

    def __init__(self, config):
        super().__init__(config)
        self._batch_boxes:list = None
        self._current_image:ndarray = None
        self._mutex = Lock()
    
    def verify(self, input_url: str) -> bool:
        check : bool = 'rstp' in input_url.lower()
        if check:
            self._input_url = input_url
        return check

    def _init_detection_network(self):
        while True:
            self._mutex.acquire()
            image = self._current_image
            self._batch_boxes = get_yolo_boxes(
                self._infer_model, 
                [image], 
                self._net_h, 
                self._net_w, 
                self._config['model']['anchors'], 
                self._obj_thresh, 
                self._nms_thresh)

    def _init_video_stream(self):

        video_reader = cv2.VideoCapture(self._input_url)
        video_reader.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ret_val, image = video_reader.read()
        image = cv2.resize(image, (1920, 1080))

        while True:
            ret_val, image = video_reader.read()
            image = cv2.resize(image, (1920, 1080))
            self._mutex.acquire()
            self._current_image = image
            draw_boxes(image, self._batch_boxes[0], self._config['model']['labels'], self._obj_thresh) 
            self._mutex.release()
            cv2.imshow('Livestream', image)

            
    
