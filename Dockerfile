FROM tacc/tacc-ml:ubuntu20.04-cuda11-tf2.6-pt1.10

LABEL maintainer="Ian Wang"

ADD src /Mirage/src
ADD script /Mirage/script

ADD requirements.txt requirements.txt
RUN pip3 --no-cache-dir install -r requirements.txt \
    && rm -f requirements.txt

WORKDIR /Mirage
ENTRYPOINT ["ls"]
