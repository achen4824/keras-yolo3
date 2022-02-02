import cv2, json
from keras import Model
from utils.utils import get_yolo_boxes
from utils.bbox import draw_boxes


#This is for rtsp video streams ex. rtsp://<username>:<pwd>@192.168.1.xxx:xxx/cam1/onvif-h264
def startRTSP(input_path:str, infer_model:Model, net_h:int, net_w:int, config:json, obj_thresh:float, nms_thresh:float) -> None:

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