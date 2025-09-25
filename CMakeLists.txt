cmake_minimum_required(VERSION 3.14)
project(DLML LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR}/include)

# ---------------------------------------------
# Set output directories for build artifacts
# ---------------------------------------------
# Place all executables in build/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Place all shared libraries in build/lib
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# -------------------------------
# Build shared library
# -------------------------------
file(GLOB_RECURSE LIB_SOURCES src/*.cpp)
file(GLOB_RECURSE HEADERS include/*.hpp include/*.h)
add_library(dlml_lib SHARED ${LIB_SOURCES} ${HEADERS})

# -------------------------------
# Build main executable
# -------------------------------
add_executable(DLML src/main.cpp)
target_link_libraries(DLML dlml_lib)

# -------------------------------
# Automatically build each test
# -------------------------------
enable_testing()
file(GLOB TEST_FILES tests/*.cpp)

foreach(TEST_FILE ${TEST_FILES})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_FILE})
    target_link_libraries(${TEST_NAME} dlml_lib)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()

# -------------------------------
# Installation rules
# -------------------------------

# Install the shared library
install(TARGETS dlml_lib
        RUNTIME DESTINATION bin        # Windows DLL
        LIBRARY DESTINATION lib        # Linux/macOS .so/.dylib
        ARCHIVE DESTINATION lib        # Static library if needed
)

# Install the main executable
install(TARGETS DLML
        RUNTIME DESTINATION bin
)

# Install headers
install(DIRECTORY include/ DESTINATION include)

# Install tests (optional)
foreach(TEST_FILE ${TEST_FILES})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    install(TARGETS ${TEST_NAME} RUNTIME DESTINATION bin/tests)
endforeach()