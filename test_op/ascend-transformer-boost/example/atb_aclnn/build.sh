#!/bin/bash

function compile_model() {
    mkdir -p build; 
    cd build;

    cmake ..;
    if [ $? -ne 0 ]; then
        echo "ERROR: generate makefile failed!"
        exit 1
    fi

    cmake --build . -j;
    if [ $? -ne 0 ]; then
        echo "ERROR: compile test failed!"
        exit 1
    else
        echo "INFO: compile test succeed!"
    fi
    cd -;

}

compile_model

