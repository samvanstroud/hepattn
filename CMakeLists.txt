cmake_minimum_required(VERSION 3.15...3.26)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

SET(CMAKE_CXX_FLAGS "-Wall -O2 -Wfatal-errors -fstrict-aliasing -std=c++17 -fopenmp -flto -march=native")
# Check if the architecture is x86_64 and append specific flags
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    message(STATUS "x86_64 platform detected, adding SSE and 64-bit flags.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -mfpmath=sse")
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_PREFIX_PATH "${pybind11_cmake_dir}")

find_package(pybind11 CONFIG REQUIRED)

python_add_library(_core MODULE src/lap1015/src/main.cpp WITH_SOABI)
target_link_libraries(_core PRIVATE pybind11::headers)
target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

add_executable(test_cpu src/lap1015/src/csrc/test/test_cpu.cpp)
# add_executable(mytest lap1015/src/csrc/test/mytest.cpp)

if (OpenMP_CXX_FOUND)
  target_link_libraries(test_cpu PUBLIC OpenMP::OpenMP_CXX)
endif (OpenMP_CXX_FOUND)

# target_link_libraries(mytest PRIVATE pybind11::headers)
# target_compile_definitions(mytest PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS _core DESTINATION lap1015)
