import imageio, requests, time, cv2, io, json, sys
from src.Stream import Stream
from lib.utils.utils import get_yolo_boxes
from lib.utils.bbox import draw_boxes
from requests.auth import HTTPDigestAuth
from requests import Session
from numpy import ndarray
from typing import Tuple

# Taken from repo https://github.com/fbchat-dev/fbchat 
# Needed some modifications for it to work though 
from lib.fbchat import Client

class HTTPStream(Stream):


    def __init__(self, config):
        super().__init__(config)
        self._authenticated : bool = False


    
    def verify(self, input_url: str) -> bool:
        check : bool = 'http' in input_url.lower()
        if check:
            self._input_url = input_url
        return check



    def authenticate(self, auth: json):
        """Set Credentials for both facebook and the webcam stream

        Args:
            auth (json): A json with the authetication details
        """
        self._auth = auth
        self._client:Client = Client(auth["facebook"]["username"], auth["facebook"]["password"])

        self._session:Session = requests.Session()
        self._session.auth = HTTPDigestAuth(auth["http"]["username"], auth["http"]["password"])

        self._authenticated:bool = True

    def _keep_login(self):
        if not self._client.isLoggedIn():
            self._client.login(self.auth["facebook"]["username"], self.auth["facebook"]["password"])


    def _sendFBMessage(self, message:str) -> None:
        """Send message to the most recent threads on messenger

        Args:
            message (str): The message to be sent
        """
        friends = self._client.fetchThreadList()
        for friend in friends:
            self._client.sendMessage(message, thread_id=friend.uid)


    
    def _sendFBFile(self, filename:str, message:str="") -> None:
        """Send file to the most recent 20 threads on messenger

        Args:
            filename (str): URI of the file to be sent
            message (str, optional): An accompanying message with the file. Defaults to "".
        """
        friends = self._client.fetchThreadList()
        for friend in friends:
            self._client.sendLocalFiles([filename], message=message, thread_id=friend.uid)



    def _dict_to_string(self, dictionary:dict) -> str:
        """Converts the dict of classes with their box count into a human-readable string

        Args:
            dictionary (dict): class to box number mapping

        Returns:
            str: Human-Readable string
        """
        result:str = ""
        keys = list(dictionary.keys())
        for key in keys:
            result += f'\n{dictionary[key]} {key}(s)' if dictionary[key] > 0 else ''
        return result            
            

    def _getImage(self) -> Tuple[bool, ndarray]:
        """Get image from IP camera often requires username and password ex. http://192.168.1.xxx/ISAPI/Streaming/channels/101/picture
        Uses the requests library

        Returns:
            Tuple[bool, ndarray]: [description]
        """
        if not self._authenticated:
            self._logger.error("Authenticate Method must be run first to setup credentials")
            sys.exit(1)

        response = self._session.get(self._input_url)
        if response.status_code == 200:
            self._logger.info(f"Successfully got image from {self._input_url}")
            image = imageio.imread(io.BytesIO(response.content))
            image = cv2.resize(image, (1920, 1080))
            return True, image
        elif response.status_codes == 404:
            self._logger.error("{self._input_url} was not found")
        elif response.status_code == 401:
            self._logger.error("{self._input_url} had bad credentials")
        else:
            self._logger.error(f"Unknown response code: {response.status_code}")
        
        return False, []

    
    def start(self):
        """Starts a continuous main loop of code to constantly check the camera
        """
        if self._input_url == None:
            self._logger.error("Failed to verify input URL before attempting to start process")
            sys.exit(1)
        
        if not self._authenticated:
            self._logger.error("Authenticate Method must be run first to setup credentials")
            sys.exit(1)

        fps:float = 1

        while True:
            image_status, image = self._getImage()
            if image_status:
                batch_boxes = get_yolo_boxes(
                    self._infer_model, 
                    [image], 
                    self._net_h, 
                    self._net_w, 
                    self._config['model']['anchors'], 
                    self._obj_thresh, 
                    self._nms_thresh)

                ret_image, box_dict = draw_boxes(image, batch_boxes[0], self._config['model']['labels'], self._obj_thresh)

                if sum(list(box_dict.values())) > 0:
                    self._keep_login()
                    filename = f'images/image-{time.strftime("%Y-%m-%d-%H:%M")}.png'
                    cv2.imwrite(filename, image)
                    self._sendFBFile(filename, message=f'{time.strftime("%Y-%m-%d-%H:%M")}{self._dict_to_string(box_dict)}')
                    time.sleep(15)
            time.sleep(1/fps)