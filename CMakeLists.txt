cmake_minimum_required(VERSION 4.2)
project(NeuralNetwork)

file(GLOB_RECURSE sources
    src/*.cpp
    src/*.h)

add_executable(NeuralNet_modif_6_ALPHA ${sources})

