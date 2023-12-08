FROM tensorflow/tensorflow:latest-gpu

COPY ./src/ /src

WORKDIR /src

RUN pip3 install matplotlib
RUN pip3 install scikit-learn

ENTRYPOINT [ "python3", "-u", "results.py", "|", "tee", "/HaGRID/latestOutput.txt"]