FROM python:3.8-slim-buster


WORKDIR /app

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip3 install tensorflow keras opencv-contrib-python imageio scipy aenum attrs bs4 paho-mqtt


COPY . .

ENV CONFIG_FILE=/app/config.json
ENV CREDENTIAL_FILE=/app/secret.json
ENV HTTP_URL=http://192.168.1.100/ISAPI/Streaming/channels/101/picture

CMD [ "python3", "main.py"]

# docker build -t person_recognition .
# docker run --rm -d --network host --name person_recog person_recognition