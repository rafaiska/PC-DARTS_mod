FROM nvidia/cuda:11.1.1-base-ubuntu20.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    python2.7 \
    git \
    curl \
 && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://bootstrap.pypa.io/pip/2.7/get-pip.py -o get-pip.py
RUN python2.7 get-pip.py

RUN pip install torch==0.3.1
RUN pip install torchvision==0.2.1
RUN ln -s /usr/bin/python2.7 /usr/bin/python

# Clone PC-DARTS
RUN mkdir workspace
RUN git clone https://github.com/yuhuixu1993/PC-DARTS.git /workspace/PC-DARTS
WORKDIR /workspace/PC-DARTS

# Set the default command to bash
CMD ["bash"]