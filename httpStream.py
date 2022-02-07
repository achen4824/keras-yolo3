import json
import imageio, requests, time, cv2, io
from utils.utils import get_yolo_boxes
from utils.bbox import draw_boxes
from keras import Model
from requests.auth import HTTPDigestAuth

# Taken from repo https://github.com/fbchat-dev/fbchat 
# Needed some modifications for it to work though 
from fbchat import Client
from fbchat.models import Message

def loginFacebook(auth : json) -> Client:
    # # facebook user credentials
    # # login
    client = Client(auth["facebook"]["username"], auth["facebook"]["password"])
    # client.sendMessage("Service Initialized", thread_id=client.uid) Send to self
    allFriends = client.fetchThreadList()
    for friend in allFriends:
        client.sendMessage("Service Initialized", thread_id=friend.uid)

    return client

# Get image from IP camera often requires username and password ex. http://192.168.1.xxx/ISAPI/Streaming/channels/101/picture
def startHTTP(input_path:str, auth : json, infer_model:Model, net_h:int, net_w:int, config:json, obj_thresh:float, nms_thresh:float) -> None:
    # the main loop
    batch_size  = 1
    images      = []

    session = requests.Session()
    session.auth = HTTPDigestAuth(auth["http"]["username"], auth["http"]["password"])
    
    # Login Facebook
    client = loginFacebook(auth)


    while True:
        response = session.get(input_path)
        arr = imageio.imread(io.BytesIO(response.content))

        image = cv2.resize(arr, (1920, 1080))

        images += [image]

        if (len(images)==batch_size) or (len(images)>0):
            batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)
            
            for i in range(len(images)):
                ret_image, num_boxes = draw_boxes(images[i], batch_boxes[i], config['model']['labels'], obj_thresh) 
                # cv2.imshow('video with bboxes', images[i])
                
                if sum(num_boxes) > 0:
                    filename = f'images/image-{time.strftime("%Y-%m-%d-%H:%M")}.png'
                    cv2.imwrite(filename, images[i])
                    allFriends = client.fetchThreadList()
                    for friend in allFriends:
                        client.sendLocalFiles([filename], message=f'{time.strftime("%Y-%m-%d-%H:%M")}\n{num_boxes[0]} person(s)\n{num_boxes[1]} dog(s)', thread_id=friend.uid)
                    time.sleep(15)

            images = []
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
        time.sleep(0.2)


