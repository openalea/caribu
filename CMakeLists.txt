cmake_minimum_required(VERSION 3.20)
project(caribu LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(CMAKE_VERBOSE_MAKEFILE ON)

include_directories("src/cpp/")

add_subdirectory("src/cpp/meschach/mesch12a/src")
add_subdirectory("src/cpp/bibliotek")
add_subdirectory("src/cpp/mc-sail/src")
add_subdirectory("src/cpp/Periodise/src")
add_subdirectory("src/cpp/s2v/src")
add_subdirectory("src/cpp/Canestra/src")

install(TARGETS mcsail periodise s2v canestrad
        DESTINATION "$ENV{PREFIX}/bin")
