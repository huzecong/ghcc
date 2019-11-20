FROM gcc:latest

# Install necessary packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    python3-dev
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py

# Credit: https://denibertovic.com/posts/handling-permissions-with-docker-volumes/
# Install `gosu` to avoid running as root.
RUN gpg --keyserver ha.pool.sks-keyservers.net --recv-keys B42F6819007F00F88E364FD4036A9C25BF357DD4
RUN curl -o /usr/local/bin/gosu -SL "https://github.com/tianon/gosu/releases/download/1.4/gosu-$(dpkg --print-architecture)" \
    && curl -o /usr/local/bin/gosu.asc -SL "https://github.com/tianon/gosu/releases/download/1.4/gosu-$(dpkg --print-architecture).asc" \
    && gpg --verify /usr/local/bin/gosu.asc \
    && rm /usr/local/bin/gosu.asc \
    && chmod +x /usr/local/bin/gosu

# Install Python libraries.
COPY requirements.txt /usr/src/
RUN pip install -r /usr/src/requirements.txt && \
    rm /usr/src/requirements.txt

# Install packages for compilation & ease-of-use.
RUN apt-get install -y --no-install-recommends \
    less \
    vim \
    bmake
#RUN apt-get install -y --no-install-recommends \
#    binutils-dev \
#    bison \
#    dialog \
#    flex \
#    flite1-dev \
#    freeglut3-dev \
#    guile-2.0-dev \
#    lib3ds-dev \
#    libaa1-dev \
#    libacl1-dev \
#    libaio-dev \
#    libao-dev \
#    libargtable2-dev \
#    libatlas-base-dev \
#    libatm1-dev \
#    libattr1-dev \
#    libaubio-dev \
#    libaudio-dev \
#    libaudit-dev \
#    libauparse-dev \
#    libavcodec-dev \
#    libavdevice-dev \
#    libavfilter-dev \
#    libavresample-dev \
#    libbam-dev \
#    libbdd-dev \
#    libbluetooth-dev \
#    libbluray-dev \
#    libboost-regex-dev \
#    libboost-system-dev \
#    libboost-thread-dev \
#    libbrlapi-dev \
#    libbs2b-dev \
#    libbtbb-dev \
#    libbwa-dev \
#    libcap-dev \
#    libcdb-dev \
#    libcdio-paranoia-dev \
#    libcfg-dev \
#    libcfitsio-dev \
#    libchewing3-dev \
#    libcjson-dev \
#    libcmap-dev \
#    libcmph-dev \
#    libcodec2-dev \
#    libcomedi-dev \
#    libconfig-dev \
#    libconfuse-dev \
#    libcpg-dev \
#    libcpupower-dev \
#    libcrack2-dev \
#    libcrmcommon-dev \
#    libcunit1-dev \
#    libcups2-dev \
#    libczmq-dev \
#    libdbi-dev \
#    libdca-dev \
#    libdebconfclient0-dev \
#    libdebian-installer4-dev \
#    libdirectfb-dev \
#    libdlm-dev \
#    libdlmcontrol-dev \
#    libdnet-dev \
#    libdts-dev \
#    libdv4-dev \
#    libdw-dev \
#    libdwarf-dev \
#    libedit-dev \
#    libenca-dev \
#    libepoxy-dev \
#    libev-dev \
#    libewf-dev \
#    libext2fs-dev \
#    libf2c2-dev \
#    libfaad-dev \
#    libfcgi-dev \
#    libfdt-dev \
#    libfftw3-dev \
#    libfiu-dev \
#    libflac-dev \
#    libfluidsynth-dev \
#    libforms-dev \
#    libfreecell-solver-dev \
#    libfreeimage-dev \
#    libfreenect-dev \
#    libftdi-dev \
#    libftdi1-dev \
#    libftgl-dev \
#    libftp-dev \
#    libfuse-dev \
#    libgadu-dev \
#    libgc-dev \
#    libgcrypt20-dev \
#    libgd-dev \
#    libgenometools0-dev \
#    libgeoip-dev \
#    libglew-dev \
#    libglfw3-dev \
#    libgnustep-base-dev \
#    libgpac-dev \
#    libgpm-dev \
#    libgps-dev \
#    libgraphicsmagick1-dev \
#    libgsl-dev \
#    libgsm1-dev \
#    libgtkdatabox-dev \
#    libharfbuzz-dev \
#    libhiredis-dev \
#    libiberty-dev \
#    libibmad-dev \
#    libibnetdisc-dev \
#    libibumad-dev \
#    libidn11-dev \
#    libigraph0-dev \
#    libiksemel-dev \
#    libimlib2-dev \
#    libimobiledevice-dev \
#    libiniparser-dev \
#    libiodbc2-dev \
#    libiptc-dev \
#    libircclient-dev \
#    libiscsi-dev \
#    libisl-dev \
#    libisns-dev \
#    libiso9660-dev \
#    libiw-dev \
#    libixp \
#    libjansson-dev \
#    libjbig2dec0-dev \
#    libjemalloc-dev \
#    libjim-dev \
#    libjson-c-dev \
#    libjudy-dev \
#    libkaz-dev \
#    libkmod-dev \
#    liblapack-dev \
#    libldap2-dev \
#    libldns-dev \
#    libleveldb-dev \
#    liblivemedia-dev \
#    liblo-dev \
#    liblua5.1-0-dev \
#    liblua5.2-dev \
#    liblua50-dev \
#    liblualib50-dev \
#    liblz4-dev \
#    libmad0-dev \
#    libmagic-dev \
#    libmarkdown2-dev \
#    libmatheval-dev \
#    libmbedtls-dev \
#    libmcrypt-dev \
#    libmd-dev \
#    libmemcached-dev \
#    libmetis-dev \
#    libmhash-dev \
#    libmicrohttpd-dev \
#    libmlt-dev \
#    libmng-dev \
#    libmnl-dev \
#    libmodbus-dev \
#    libmodplug-dev \
#    libmowgli-2-dev \
#    libmp3lame-dev \
#    libmpc-dev \
#    libmpcdec-dev \
#    libmpfr-dev \
#    libmpg123-dev \
#    libmtp-dev \
#    libmunge-dev \
#    libneon27-dev \
#    libnet1-dev \
#    libnetcdf-dev \
#    libnetfilter-conntrack-dev \
#    libnetfilter-queue-dev \
#    libnetpbm10-dev \
#    libnewt-dev \
#    libnfnetlink-dev \
#    libnids-dev \
#    libnl-genl-3-dev \
#    libnl-nf-3-dev \
#    libnlopt-dev \
#    libnotify-dev \
#    liboauth-dev \
#    libopenal-dev \
#    libopencc-dev \
#    libopencore-amrnb-dev \
#    libopencore-amrwb-dev \
#    libopencv-core-dev \
#    libopencv-flann-dev \
#    libopencv-imgproc-dev \
#    libopenr2-dev \
#    libosip2-dev \
#    libpam0g-dev \
#    libpapi-dev \
#    libparted-dev \
#    libpcap-dev \
#    libpci-dev \
#    libpciaccess-dev \
#    libpcl1-dev \
#    libpcp-pmda3-dev \
#    libpcp3-dev \
#    libpcsclite-dev \
#    libperl-dev \
#    libpfm4-dev \
#    libpopt-dev \
#    libportmidi-dev \
#    libpri-dev \
#    libproj-dev \
#    libpsl-dev \
#    libpth-dev \
#    libpython2.7-dev \
#    libqrencode-dev \
#    libquicktime-dev \
#    libquorum-dev \
#    librabbitmq-dev \
#    librados-dev \
#    librbd-dev \
#    librdkafka-dev \
#    librdmacm-dev \
#    librrd-dev \
#    librtmp-dev \
#    libs3-dev \
#    libsamplerate0-dev \
#    libsasl2-dev \
#    libsctp-dev \
#    libsdl-gfx1.2-dev \
#    libsdl-image1.2-dev \
#    libsdl-mixer1.2-dev \
#    libsdl-ttf2.0-dev \
#    libsdl2-mixer-dev \
#    libsdl2-net-dev \
#    libsgutils2-dev \
#    libshout3-dev \
#    libsigsegv-dev \
#    libsmbclient-dev \
#    libsmi2-dev \
#    libsnappy-dev \
#    libsndfile1-dev \
#    libsndio-dev \
#    libsoil-dev \
#    libspandsp-dev \
#    libspectrum-dev \
#    libspeex-dev \
#    libspeexdsp-dev \
#    libspiro-dev \
#    libsprng2-dev \
#    libsqlite0-dev \
#    libss7-dev \
#    libssh-dev \
#    libssh2-1-dev \
#    libst-dev \
#    libstrophe-dev \
#    libswresample-dev \
#    libswscale-dev \
#    libsysfs-dev \
#    libtalloc-dev \
#    libtar-dev \
#    libtcc-dev \
#    libtcl8.6 \
#    libtdb-dev \
#    libtheora-dev \
#    libtokyocabinet-dev \
#    libtokyotyrant-dev \
#    libtommath-dev \
#    libtonezone-dev \
#    libtpm-unseal-dev \
#    libtrace3-dev \
#    libtre-dev \
#    libtrio-dev \
#    libtspi-dev \
#    libtwolame-dev \
#    libucl-dev \
#    libunbound-dev \
#    liburcu-dev \
#    libusb-1.0-0-dev \
#    libusb-dev \
#    libvdeplug-dev \
#    libvdpau-dev \
#    libvirt-dev \
#    libvncserver-dev \
#    libvo-amrwbenc-dev \
#    libwavpack-dev \
#    libwbclient-dev \
#    libwebsockets-dev \
#    libwrap0-dev \
#    libx264-dev \
#    libxaw7-dev \
#    libxcb-icccm4-dev \
#    libxcb-xinerama0-dev \
#    libxerces-c-dev \
#    libxft-dev \
#    libxi-dev \
#    libxmltok1-dev \
#    libxmu-dev \
#    libxnvctrl-dev \
#    libxosd-dev \
#    libxpm-dev \
#    libxtables-dev \
#    libxtst-dev \
#    libxvidcore-dev \
#    libxxf86dga-dev \
#    libxxhash-dev \
#    libzdb-dev \
#    libzip-dev \
#    libzmq3-dev \
#    nasm \
#    ocl-icd-opencl-dev \
#    portaudio19-dev \
#    vstream-client-dev

# Create entrypoint that sets appropriate group and user IDs.
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Copy `ghcc` files into image, and set PYTHONPATH and PATH.
ENV CUSTOM_PATH="/usr/custom"
COPY ghcc/ $CUSTOM_PATH/ghcc/
COPY scripts/ $CUSTOM_PATH/scripts/
RUN curl -sSL https://github.com/shyiko/commacd/raw/v1.0.0/commacd.sh -o $CUSTOM_PATH/.commacd.sh
ENV PATH="$CUSTOM_PATH/scripts/mock_path:$PATH"
ENV PYTHONPATH="$CUSTOM_PATH/:$PYTHONPATH"
