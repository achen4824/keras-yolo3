#! /usr/bin/env python
import os
import time
import argparse
import json
import io
import imageio
import cv2
import requests
from requests.auth import HTTPDigestAuth
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
from keras.models import load_model
from tqdm import tqdm
import numpy as np

def _main_(args):
    config_path  = args.conf
    input_path   = args.input
    output_path  = args.output

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    makedirs(output_path)

    ###############################
    #   Set some parameter
    ###############################       
    net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.9, 0.45

    ###############################
    #   Load the model
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    infer_model = load_model(config['train']['saved_weights_name'], compile=False)

    ###############################
    #   Predict bounding boxes 
    ###############################
    if 'rtsp' in input_path: # do detection on the first webcam
        video_reader = cv2.VideoCapture(input_path)
        # video_reader.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        # video_reader.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
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
    elif 'http' in input_path:

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


        
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--input', help='path to an image, a directory of images, a video, or webcam')    
    argparser.add_argument('-o', '--output', default='output/', help='path to output directory')   
    
    args = argparser.parse_args()
    _main_(args)
