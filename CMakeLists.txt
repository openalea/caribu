cmake_minimum_required(VERSION 3.20)
project(caribu LANGUAGES C CXX)

if(DEFINED ENV{SKIP_BUILD})
  message(STATUS "Skipping build of C/C++ extensions")
  return()  # stops processing the CMakeLists here
endif()

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(CMAKE_VERBOSE_MAKEFILE ON)

# Needed for Macosx-latest numerical stability
if (APPLE)
    add_compile_options(-ffp-contract=off)
endif()

include_directories("src/cpp/")

add_subdirectory("src/cpp/meschach/mesch12a/src")
add_subdirectory("src/cpp/bibliotek")
add_subdirectory("src/cpp/mc-sail/src")
add_subdirectory("src/cpp/Periodise/src")
add_subdirectory("src/cpp/s2v/src")
add_subdirectory("src/cpp/Canestra/src")

install(TARGETS mcsail periodise s2v canestrad
        DESTINATION bin)

