
cmake_minimum_required(VERSION 3.16)

project(PBL6_Embedded LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(oled_temp
    src/oled_temp.cpp
    src/oled.cpp
    src/system_info.cpp
)

target_include_directories(oled_temp PRIVATE include)

target_compile_options(oled_temp PRIVATE -Wall -Wextra -O2)
