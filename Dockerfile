FROM eywalker/pytorch-jupyter:v0.4.0-updated
    
WORKDIR /src

RUN apt-get -y update && apt-get  -y install ffmpeg libhdf5-10
RUN pip3 install imageio ffmpy h5py opencv-python statsmodels

RUN git clone https://github.com/atlab/attorch.git && \
    pip install -e attorch/

RUN git clone https://github.com/cajal/neuro_data.git && \
    pip install -e neuro_data/

#RUN pip install --upgrade git+https://github.com/datajoint/datajoint-python.git

ADD . /src/static-networks
RUN pip3 install -e /src/static-networks

WORKDIR /notebooks


