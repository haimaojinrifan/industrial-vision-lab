cmake_minimum_required(VERSION 3.21)
project(industrial_vision_lab CXX)

# ---------------------------------------------------------------------------
# Build configuration
# ---------------------------------------------------------------------------
# vcpkg ships a Debug build of OpenCV. Mixing that Debug DLL with a Release
# executable crashes inside cvtColor (different CRT / different SIMD paths),
# which shows up as a grey window instead of an image. Map the Debug config of
# the imported OpenCV targets onto RELEASE so our code keeps its debug info
# while OpenCV is always the working Release build.
set(CMAKE_MAP_IMPORTED_CONFIG_DEBUG RELEASE)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_PROBES "Build the one-off diagnostic probes in tools/" OFF)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV ${OpenCV_VERSION} from ${OpenCV_DIR}")

# ---------------------------------------------------------------------------
# add_cv_app(<target> <sources...>)
#   Links OpenCV and, on Windows, copies the runtime DLLs next to the exe so
#   the program runs without touching PATH.
# ---------------------------------------------------------------------------
function(add_cv_app name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE ${OpenCV_LIBS})

    if(WIN32)
        set(_runtime_dirs)
        if(OpenCV_INSTALL_PATH)
            list(APPEND _runtime_dirs "${OpenCV_INSTALL_PATH}/bin"
                                      "${OpenCV_INSTALL_PATH}/debug/bin")
        endif()
        get_filename_component(_compiler_bin_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
        list(APPEND _runtime_dirs "${_compiler_bin_dir}")

        add_custom_command(TARGET ${name} POST_BUILD
            COMMAND ${CMAKE_COMMAND}
                    -DEXE=$<TARGET_FILE:${name}>
                    -DDEST=$<TARGET_FILE_DIR:${name}>
                    "-DSEARCH_DIRS=$<JOIN:${_runtime_dirs},|>"
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/copy_runtime_dlls.cmake"
            COMMENT "Copying runtime DLLs for ${name}"
            VERBATIM)
    endif()
endfunction()

# ---------------------------------------------------------------------------
# Main programs
# ---------------------------------------------------------------------------
add_cv_app(day1_hello     src/day1_hello.cpp)
add_cv_app(day2_mat       src/day2_mat.cpp)
add_cv_app(day3_cap       src/day3_cap.cpp)
add_cv_app(day4_calib     src/day4_calib.cpp)
add_cv_app(day4_check     src/day4_check.cpp)
add_cv_app(day4_straight  src/day4_straight.cpp)
add_cv_app(day5_detect    src/day5_detect.cpp)
add_cv_app(day6_measure   src/day6_measure.cpp)
add_cv_app(day8_homography src/day8_homography.cpp)

# Synthetic data generators (they do not need a camera)
add_cv_app(gen_workpiece  tools/gen_workpiece.cpp)
add_cv_app(gen_circle     tools/gen_circle.cpp)

# day6_measure uses std::filesystem -> needs C++17
set_target_properties(day6_measure day8_homography PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Diagnostic probes (optional)
#   Each probe_*.cpp is a one-off experiment that located one specific bug.
#   They are kept in the repo as a record of the debugging process.
#   Configure with -DBUILD_PROBES=ON to build them.
# ---------------------------------------------------------------------------
if(BUILD_PROBES)
    set(_probes
        probe_blur probe_cam probe_cam_fps
        probe_d5 probe_d5b probe_d5c probe_d5d probe_d5e
        probe_d6 probe_exp3 probe_exposure probe_exposure2
        probe_gn probe_model probe_radial
    )
    foreach(_p IN LISTS _probes)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/${_p}.cpp")
            add_cv_app(${_p} tools/${_p}.cpp)
            set_target_properties(${_p} PROPERTIES CXX_STANDARD 17)
        endif()
    endforeach()
endif()

