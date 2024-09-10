FROM python:3.11-slim

COPY ./requirements.txt ./requirements.txt

RUN apt-get update
RUN apt-get -y install g++

RUN pip install wheel setuptools pip --upgrade
RUN pip install swig

RUN pip install -r requirements.txt

WORKDIR /code

COPY ./src ./src

EXPOSE 8080

CMD exec uvicorn src.main:app --host 0.0.0.0 --port 8080