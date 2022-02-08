from keras import Model
from keras.models import load_model
import logging

# Class interface for the other class extensions besides the initialization the other methods
# are just for display
class Stream:

    def __init__(self, config):
        self._infer_model:Model = load_model(config['train']['saved_weights_name'], compile=False)
        self._net_h = 416 
        self._net_w = 416
        self._obj_thresh = 0.95
        self._nms_thresh = 0.5
        self._config = config


        # Create logger
        self._logger = logging.getLogger('HTTPStream')
        self._logger.setLevel(logging.ERROR)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'))
        self._logger.addHandler(handler)
    
    def verify(self, input_str : str) -> bool:
        return False

    def start(self) -> None:
        return