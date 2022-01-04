# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.13)
project(onnxrt_test)
set(CMAKE_CXX_STANDARD 11)

add_executable(fns_candy_style_transfer "fns_candy_style_transfer.c")
target_include_directories(fns_candy_style_transfer PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(fns_candy_style_transfer PRIVATE onnxruntime png)
if(PNG_LIBDIR)
  target_link_directories(fns_candy_style_transfer PRIVATE ${PNG_LIBDIR})
endif()
