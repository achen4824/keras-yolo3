#! /usr/bin/env python
import os
import argparse
import json
from src.HTTPStream import HTTPStream


def _main_():
    config_path  = os.environ.get("CONFIG_FILE")
    auth_path    = os.environ.get("CREDENTIAL_FILE")
    input_path   = os.environ.get("HTTP_URL")

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    with open(auth_path) as auth_buffer:
        auth = json.load(auth_buffer)

    # Initialize Model from Saved file
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']

    stream = HTTPStream(config)
    stream.verify(input_path)
    stream.authenticate(auth)
    stream.start()
        
if __name__ == '__main__':
    _main_()
