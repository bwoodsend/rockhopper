# Docker image generator for manylinux. See:
# https://github.com/pypa/manylinux/tree/manylinux1

# Build with: (note the -t rockhopper-manylinux1_x86_64 is just an arbitrary name)
#  $ docker build -t rockhopper-manylinux1_x86_64 .
# Or to specify an alternative base (say manylinux1_i686 for 32bit Linux):
#  $ docker build -t rockhopper-manylinux1_i686 --build-arg BASE=manylinux1_i686 .

# Then boot into your new image with:
#  $ docker run -v `pwd`:/io -it rockhopper-manylinux1_x86_64
# The above launches bash inside the image. You can append arbitrary shell
# commands to run those instead such as the following to launch Python:
#  $ docker run -v `pwd`:/io -it rockhopper-manylinux1_x86_64 python
# Or to run pytest:
#  $ docker run -v `pwd`:/io -it rockhopper-manylinux1_x86_64 pytest

ARG BASE=manylinux1_x86_64
FROM quay.io/pypa/${BASE}

# Choosing a Python version is done just by prepending its bin dir to PATH.
ENV PATH=/opt/python/cp310-cp310/bin:/opt/python/cp39-cp39/bin:$PATH

# Install dependencies. Do this before COPY to encourage caching.
RUN pip install --prefer-binary wheel auditwheel numpy
RUN pip install cslug coverage toml

# Copy across enough of this repo to build from.
RUN mkdir -p /io/rockhopper /io/tests
COPY tests/requirements.txt /io/tests
COPY setup.py /io
COPY pyproject.toml /io
COPY README.rst /io
COPY rockhopper/_version.py /io/rockhopper
# The rest is expected to be -v mounted at runtime.

# Set this repo's root as the cwd.
WORKDIR /io

# Install it. Skip build isolation for speed (and sanity).
RUN pip install --prefer-binary --no-build-isolation -e . -r tests/requirements.txt
