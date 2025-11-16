cmake_minimum_required(VERSION 4.2)
project(NeuralNetwork)

file(GLOB_RECURSE sources
    src/*.cpp
    src/*.h)

add_executable(NeuralNet_modif_4 ${sources})
target_compile_definitions(NeuralNet_modif_4 PRIVATE TRAIN_DIR="${CMAKE_CURRENT_LIST_DIR}/train")

