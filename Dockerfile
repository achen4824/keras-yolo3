FROM python:3.8-slim-buster


WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

ENV CONFIG_FILE=/app/config.json
ENV CREDENTIAL_FILE=/app/secret.json
ENV HTTP_URL=http://192.168.1.100/ISAPI/Streaming/channels/101/picture

CMD [ "python3", "main.py"]