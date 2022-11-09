FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get upgrade -y 
RUN apt-get install -y \
               wget \
               unzip \
               ssh \
               git \
               vim
RUN apt-get clean

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

WORKDIR /fact-linking
COPY requirements.txt /fact-linking

RUN python -m pip install --upgrade pip
RUN pip install sentence-transformers
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
RUN pip install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
RUN pip install torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
RUN pip install -r requirements.txt
RUN pip cache purge

RUN groupadd -g 76084 nlplogins_AppGrpU
RUN groupadd -g 11131 NLP-StaffU
RUN useradd -rm -d /home/postuvan -s /bin/bash -g 76084 -G sudo,NLP-StaffU -u 245024 postuvan
USER postuvan