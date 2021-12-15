FROM carlasim/carla:0.9.11

USER root
RUN apt-get update
RUN apt-get install -y software-properties-common &&\
        apt-add-repository universe &&\
            apt-get update &&\
            apt-get install -y python3-pip
RUN apt-get install -y libpng16-16 libtiff5 libjpeg-turbo8 wget xdg-user-dirs xdg-utils xvfb rsync && rm -rf /var/lib/apt/lists/*

RUN chmod -R 777 /root
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

RUN pip install pygame networkx psutil numba torch imageio numpy opencv-python
COPY . /home/carla_dataset
WORKDIR /home/carla_dataset

ENV CARLAPATH /home/carla/PythonAPI/carla/
RUN chmod -R 777 .

USER carla

ENTRYPOINT ["./docker_run_collection.sh"]
