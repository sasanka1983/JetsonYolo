FROM nvcr.io/nvidia/l4t-ml:r32.7.1-py3

WORKDIR /opt

RUN apt-get update && apt-get install -y python3-tk

RUN pip3 install tqdm

RUN apt-get clean

COPY . .

CMD ["python3","JetsonYolo.py"]
