
FROM ubuntu:latest AS  build

RUN  apt update && apt update -y
RUN  apt-get install -y python3
RUN  apt-get -y install python3-pip

#From python:3.9


WORKDIR micro0

ARG profile=dev
ENV environment=${profile}

COPY main.py main.py
COPY src src
COPY requirements.txt requirements.txt
#COPY properties properties

RUN pip install -r requirements.txt
#RUN python3 -m spacy download en_core_web_sm




EXPOSE 8004
CMD gunicorn -w 1 -k uvicorn.workers.UvicornWorker --env environment=${environment} main:app -b 0.0.0.0:8004
#CMD  ["uvicorn","main:app","--host", "0.0.0.0","--port","8004"]
