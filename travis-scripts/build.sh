#!/bin/bash

CC=gcc-7 && CXX=g++-7

mkdir build
cd build
cmake ..
make unit_tests gui
cd ../unit_tests_data
../build/unit_tests/unit_tests
