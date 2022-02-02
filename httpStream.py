import json
import imageio, requests, time, cv2, io
from utils.utils import get_yolo_boxes
from utils.bbox import draw_boxes
from keras import Model
from requests.auth import HTTPDigestAuth

def startHTTP(input_path:str, infer_model:Model, net_h:int, net_w:int, config:json, obj_thresh:float, nms_thresh:float) -> None:
    # the main loop
    batch_size  = 1
    images      = []

    session = requests.Session()
    session.auth = HTTPDigestAuth(username, password)


    while True:
        response = session.get(input_path)
        arr = imageio.imread(io.BytesIO(response.content))

        image = cv2.resize(arr, (1920, 1080))

        images += [image]

        if (len(images)==batch_size) or (len(images)>0):
            batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)

            for i in range(len(images)):
                draw_boxes(images[i], batch_boxes[i], config['model']['labels'], obj_thresh) 
                cv2.imshow('video with bboxes', images[i])
            images = []
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
        time.sleep(0.2)