FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN mkdir /app
ADD req2.txt /app
RUN pip install -r /app/req2.txt
#RUN conda install --file /app/req2.txt
