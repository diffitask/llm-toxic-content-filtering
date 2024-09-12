FROM python:3.11-slim

COPY ./requirements.txt ./requirements.txt

RUN apt-get update
RUN apt-get -y install g++

RUN pip install wheel setuptools pip --upgrade
RUN pip install swig

RUN pip install -r requirements.txt

EXPOSE 8080