#! /usr/bin/env python
import os
import argparse
import json
from keras.models import load_model
from httpStream import startHTTP
from rtspStream import startRTSP

def _main_(args):
    config_path  = args.conf
    input_path   = args.input
    username     = args.username
    password     = args.password

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)


    # Yolov3 Parameters  
    net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.9, 0.45
    

    # Initialize Model from Saved file
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    infer_model = load_model(config['train']['saved_weights_name'], compile=False)

    
    if 'rtsp' in input_path:
        startRTSP(input_path, infer_model, net_h, net_w, config, obj_thresh, nms_thresh)
    elif 'http' in input_path:
        if username == None or password == None:
            print('HTTP Stream requires a username or password')
            return
        startHTTP(input_path, username, password, infer_model, net_h, net_w, config, obj_thresh, nms_thresh)



        
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--input', help='path to rtsp or http stream')    
    argparser.add_argument('-u', '--username', help='username for http stream', required=False)
    argparser.add_argument('-p', '--password', help='password for http stream', required=False)

    args = argparser.parse_args()
    _main_(args)
