cmake_minimum_required(VERSION 3.14)
project(NovaML LANGUAGES CXX)

# -------------------------------
# C++ standard
# -------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # needed for shared library

# -------------------------------
# Include directories
# -------------------------------
include_directories(${PROJECT_SOURCE_DIR}/include)

# -------------------------------
# Output directories
# -------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# -------------------------------
# Header-only modules (Core, Data, Trainer)
# -------------------------------
file(GLOB_RECURSE HEADER_ONLY
    ${PROJECT_SOURCE_DIR}/include/NovaML/**/*.hpp
    ${PROJECT_SOURCE_DIR}/include/NovaML/**/*.tpp
)

# -------------------------------
# Compiled modules: Parallel + Utils + optional Trainer helpers
# -------------------------------
file(GLOB_RECURSE COMPILED_SOURCES
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

# -------------------------------
# Build shared library
# -------------------------------
add_library(novaml_lib SHARED ${COMPILED_SOURCES} ${HEADER_ONLY})

# Optional: enable OpenMP if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(novaml_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

# -------------------------------
# Build main executable
# -------------------------------
add_executable(NovaML src/main.cpp ${HEADER_ONLY})
target_link_libraries(NovaML novaml_lib)

# -------------------------------
# Automatically build and register tests
# -------------------------------
enable_testing()
file(GLOB_RECURSE TEST_SOURCES ${PROJECT_SOURCE_DIR}/tests/*.cpp)

foreach(TEST_FILE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_FILE} ${HEADER_ONLY})
    target_link_libraries(${TEST_NAME} novaml_lib)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()

# -------------------------------
# Installation rules
# -------------------------------
install(TARGETS novaml_lib
        RUNTIME DESTINATION bin        # Windows DLL
        LIBRARY DESTINATION lib        # Linux/macOS .so/.dylib
        ARCHIVE DESTINATION lib)       # Static library if needed

install(TARGETS NovaML
        RUNTIME DESTINATION bin)

install(DIRECTORY include/ DESTINATION include)

foreach(TEST_FILE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    install(TARGETS ${TEST_NAME} RUNTIME DESTINATION bin/tests)
endforeach()