FROM tacc/tacc-ml:ubuntu20.04-cuda11-tf2.6-pt1.10

LABEL maintainer="Ian Wang"

ADD offline_data_gen /pro_2/offline_data_gen
ADD online_validation /pro_2/online_validation
ADD train_model /pro_2/train_model

ADD requirements.txt requirements.txt
RUN pip3 --no-cache-dir install -r requirements.txt \
    && rm -f requirements.txt

WORKDIR /pro_2
ENTRYPOINT ["ls"]
