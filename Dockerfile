# https://hub.docker.com/r/pytorch/pytorch
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
# NOTE:
# pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime has python version 3.10.8, system version Ubuntu 18.04.6 LTS
# pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime has python version 3.7.x
# pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime has python version 3.10.11, system version Ubuntu 20.04.6 LTS
# pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime has python version 3.10.13, system version Ubuntu 20.04.6 LTS
# pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime has python version 3.10.13, system version Ubuntu 22.04.3 LTS
# pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime has python version 3.10.14, system version Ubuntu 22.04.4 LTS


## The MAINTAINER instruction sets the author field of the generated images.
LABEL maintainer="wenh06@gmail.com"


# set the environment variable to avoid interactive installation
# which might stuck the docker build process
ENV DEBIAN_FRONTEND=noninteractive

ENV HUGGINGFACE_HUB_CACHE=/challenge/cache/revenger_model_dir
ENV HF_HUB_CACHE=/challenge/cache/revenger_model_dir
# ENV MODEL_CACHE_DIR=/challenge/cache/revenger_model_dir
ENV MODEL_CACHE_DIR=/challenge/save_pths/val-test.pth
ENV GIT_CLONE_DIR=/challenge/cache/git_clone_dir

ENV INPUT_DIR=/input
ENV OUTPUT_DIR=/output

ENV NO_ALBUMENTATIONS_UPDATE=1
ENV ALBUMENTATIONS_DISABLE_VERSION_CHECK=1

ENV TF_CPP_MIN_LOG_LEVEL=2


RUN mkdir -p $INPUT_DIR $OUTPUT_DIR


# check distribution of the base image
RUN cat /etc/issue

# check detailed system version of the base image
RUN cat /etc/os-release

# check python version of the base image
RUN python --version

# check CUDA version of the base image if is installed
RUN if [ -x "$(command -v nvcc)" ]; then nvcc --version; fi


# latest version of biosppy uses opencv
# https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt update
RUN apt install build-essential -y
RUN apt install git ffmpeg libsm6 libxext6 vim libsndfile1 libxrender1 unzip -y


RUN mkdir /challenge
COPY ./requirements-docker.txt /challenge
WORKDIR /challenge


RUN mkdir -p $MODEL_CACHE_DIR
RUN mkdir -p $GIT_CLONE_DIR


# RUN ln -s /usr/bin/python3 /usr/bin/python && ln -s /usr/bin/pip3 /usr/bin/pip
RUN which python

# list packages installed in the base image
RUN pip list

# torch and related packages (torchvision, torchaudio, etc.) are already installed in the base image


RUN python -m pip install --upgrade pip setuptools wheel build

# RUN pip install torch-ecg
# install the dev branch of torch-ecg
RUN cd $GIT_CLONE_DIR \
    && git clone https://github.com/DeepPSP/torch_ecg.git && cd torch_ecg && git checkout dev \
    && python -m pip install -r requirements.txt && python -m pip install -e .[dev] \
    && cd /challenge

# install dependencies other than torch-related packages
RUN pip install -r requirements-docker.txt

# list packages after installing requirements
RUN pip list

# copy the whole project to the docker container
COPY ./ /challenge

# Download pretrained models
RUN python post_docker_build.py
# check if the data and model are downloaded
# TODO: pass the path as environment variables
RUN du -sh $INPUT_DIR
RUN du -sh $MODEL_CACHE_DIR


## Execute the inference command
CMD ["predict.py"]
ENTRYPOINT ["python3"]
