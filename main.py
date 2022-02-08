#! /usr/bin/env python
import os
import argparse
import json
from keras.models import load_model
from src.HTTPStream import startHTTP
from src.RTSPStream import startRTSP


def _main_(args):
    config_path  = args.conf
    auth_path    = args.auth
    input_path   = args.input

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    with open(auth_path) as auth_buffer:
        auth = json.load(auth_buffer)
    
    # Initialize Model from Saved file
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']

    if 'rtsp' in input_path:
        startRTSP(input_path)
    elif 'http' in input_path:
        if auth == None:
            print('HTTP Stream requires a username or password')
            return
        startHTTP(input_path, auth,  config)
        
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--input', help='path to rtsp or http stream')    
    argparser.add_argument('-a', '--auth', help='authetication file', required=False)


    args = argparser.parse_args()
    _main_(args)
