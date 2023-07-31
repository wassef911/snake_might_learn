FROM python:3.11-slim-buster

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update
RUN apt-get upgrade -y

RUN pip install --upgrade pip
COPY consumer/requirements.txt .

RUN pip install -r requirements.txt

COPY ./ .