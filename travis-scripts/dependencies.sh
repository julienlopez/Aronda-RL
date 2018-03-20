#!/bin/bash

git clone https://github.com/Microsoft/GSL
sudo ln -s $PWD/GSL/include/gsl /usr/include

sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update -qq
sudo apt-get update
sudo apt-get install -y cmake libboost-all-dev gcc-7 g++-7 lcov libeigen3-dev

sudo update-alternatives --install /usr/bin/gcov gcov /usr/bin/gcov-7 90
