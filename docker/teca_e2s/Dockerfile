# Use TECA's prerequisites to build a Docker image for TECA's GPU version
FROM asdufek/teca_prereqs:gpu

# *********************
# set build parameters
# *********************
ARG TECA_BRANCH=tc-detector-gpu-version
ARG CMAKE_BUILD_TYPE=Debug

ARG HOME=/home/$USER
ARG INSTALL_PREFIX=/usr/local
ARG TECA_INSTALL_PREFIX=$INSTALL_PREFIX
ARG TECA_BUILD_ROOT=/TECA_build
ARG TECA_SRC_ROOT=/TECA
ENV TECA_DATA_DIR=/TECA_data
ARG TECA_HYPERTHREADS_PER_CORE=2
ARG REQUIRE_NETCDF_MPI=TRUE

ARG TELITE_INSTALL_PREFIX=$INSTALL_PREFIX/TELite
ARG TELITE_BUILD_ROOT=/TELite_build
ARG TELITE_SRC_ROOT=/TELite

# pull down TELite
RUN git clone https://github.com/LBL-EESA/TELite.git $TELITE_SRC_ROOT && \
    cd $TELITE_SRC_ROOT && \
    mkdir -p $TELITE_BUILD_ROOT && cd $TELITE_BUILD_ROOT && \
    cmake -DCMAKE_INSTALL_PREFIX=$TELITE_INSTALL_PREFIX $TELITE_SRC_ROOT && \
    make -j && make -j install

# pull down the latest version of TECA
RUN git clone --recursive https://github.com/LBL-EESA/TECA.git $TECA_SRC_ROOT && \
    cd $TECA_SRC_ROOT && git checkout $TECA_BRANCH

RUN find $TECA_SRC_ROOT -type f \( -name "*.cxx" -o -name "*.h" \) -exec sed -i \
's/: public thrust::unary_function<[^>]*>//g' {} +

RUN mkdir -p $TECA_BUILD_ROOT && cd $TECA_BUILD_ROOT && \
    cmake \
        -Wno-dev \
        -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
        -DENABLE_MPI=ON \
        -DBUILD_TESTING=ON \
        -DTECA_ENABLE_PROFILER=ON \
        -DTECA_DATA_ROOT=$TECA_DATA_DIR \
        -DTECA_TEST_CORES=16 \
        -DHYPERTHREADS_PER_CORE=$TECA_HYPERTHREADS_PER_CORE \
        -DREQUIRE_OPENSSL=TRUE \
        -DREQUIRE_BOOST=TRUE \
        -DREQUIRE_NETCDF=TRUE \
        -DREQUIRE_NETCDF_MPI=$REQUIRE_NETCDF_MPI \
        -DREQUIRE_UDUNITS=TRUE \
        -DREQUIRE_MPI=TRUE \
        -DREQUIRE_PYTHON=TRUE \
        -DREQUIRE_TECA_DATA=FALSE \
        -DREQUIRE_CUDA=TRUE \
        -DCMAKE_INSTALL_PREFIX=$TECA_INSTALL_PREFIX \
        -DTELite_DIR=$TELITE_INSTALL_PREFIX/lib \
        $TECA_SRC_ROOT && \
    make -j 16 install && \
    cp -r $TECA_BUILD_ROOT/bin /tmp && cp -r $TECA_BUILD_ROOT/lib /tmp && \
    make clean && rm -rf $TECA_BUILD_ROOT/bin && rm -rf $TECA_BUILD_ROOT/lib && \
    mv /tmp/bin $TECA_BUILD_ROOT/ && mv /tmp/lib $TECA_BUILD_ROOT/

# make sure TECA binaries and python packages are accessible
ENV LD_LIBRARY_PATH=$INSTALL_PREFIX/lib:$INSTALL_PREFIX/lib64:$TECA_INSTALL_PREFIX/lib/:$TECA_INSTALL_PREFIX/lib64/:$LD_LIBRARY_PATH \
    PKG_CONFIG_PATH=$INSTALL_PREFIX/lib/pkgconfig:$INSTALL_PREFIX/lib64/pkgconfig:$TECA_INSTALL_PREFIX/lib/pkgconfig:$TECA_INSTALL_PREFIX/lib64/pkgconfig \
    PYTHONPATH=$TECA_INSTALL_PREFIX/lib \
    PATH=$INSTALL_PREFIX/bin/:$TECA_INSTALL_PREFIX/bin/:$TECA_BUILD_ROOT/bin/test/:$PATH

ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9+PTX"

# install earth2studio
ENV EARTH2STUDIO_PACKAGE_TIMEOUT=2000
ENV EARTH2STUDIO_CACHE=/tmp/earth2studio_cache
RUN pip install --break-system-packages "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git@0.12.0"
RUN pip install --break-system-packages  earth2studio[data]
RUN pip install --break-system-packages  earth2studio[perturbation]
RUN pip install --break-system-packages  earth2studio[cyclone]
ENV FORCE_CUDA_EXTENSION=1
RUN pip install --break-system-packages --no-build-isolation torch-harmonics==0.8.0
RUN pip install --break-system-packages "makani @ git+https://github.com/NVIDIA/modulus-makani.git@28f38e3e929ed1303476518552c64673bbd6f722"
RUN pip uninstall --break-system-packages -y nvidia-physicsnemo
RUN pip install --break-system-packages nvidia-physicsnemo==1.3.0
RUN pip install --break-system-packages earth2studio[fcn3]

ENV E2S_DX=/usr/local/lib/python3.12/dist-packages/earth2studio/models/dx/

COPY Tracking_functions.py aews_detect.py tempest_tc_detect.py C_circle_functions.c teca_ar_detect.py $E2S_DX

RUN cc -shared -Wl,-soname,C_circle_functions \
       -fopenmp -fPIC \
       -o $E2S_DX/C_circle_functions.so \
       $E2S_DX/C_circle_functions.c
