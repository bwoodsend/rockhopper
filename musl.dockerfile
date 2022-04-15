arg BASE=musllinux_1_1_x86_64
from quay.io/pypa/${BASE}

RUN apk add py3-numpy py3-pip py3-wheel py3-toml
RUN pip3 install cslug
COPY . /io
WORKDIR /io
