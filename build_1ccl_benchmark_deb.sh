#! /bin/bash

deb_version=$1

if [ ! -d "build/_install" ]; then
    echo "build/_install not exist,please make install first!"
    exit 1
fi

if [ -z "$deb_version" ]; then
    echo "Please input deb version!"
    exit 1
fi

if [ -d "1ccl_benchmark_deb_release" ]; then
    rm -rf 1ccl_benchmark_deb_release
fi
cp -rd 1ccl_benchmark_deb_src 1ccl_benchmark_deb_release

cd 1ccl_benchmark_deb_release
cp ../build/_install/examples/benchmark/benchmark extract/opt/oneccl_benchmark_custom/benchmark/1ccl_benchmark

sed -i "s/oneccl_version_replace/$deb_version/g" extract/DEBIAN/control
sed -i "s/oneccl_version_replace/$deb_version/g" extract/DEBIAN/postinst
sed -i "s/oneccl_version_replace/$deb_version/g" extract/DEBIAN/preinst
sed -i "s/oneccl_version_replace/$deb_version/g" extract/DEBIAN/postrm

mkdir build
dpkg-deb -b extract/ build/

