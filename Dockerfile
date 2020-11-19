FROM ubuntu:latest

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN cd /usr/local/bin \
  && ln -s /usr/bin/python3 python
RUN pip3 install torch transformers==3.0.2 numpy pandas flask gevent gunicorn contractions
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY distilbert /opt/program
WORKDIR /opt/program

