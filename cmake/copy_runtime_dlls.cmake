# Copy the runtime DLLs that <EXE> depends on (OpenCV, libstdc++, ...) into DEST.
#
# Called from CMakeLists.txt with:
#   EXE          full path of the executable
#   DEST         its output directory
#   SEARCH_DIRS  DLL search directories, separated by |

if(POLICY CMP0207)
    cmake_policy(SET CMP0207 NEW)
endif()

if(NOT EXISTS "${EXE}")
    message(FATAL_ERROR "executable not found: ${EXE}")
endif()

string(REPLACE "|" ";" SEARCH_DIRS "${SEARCH_DIRS}")

set(_search_dirs)
foreach(_dir IN LISTS SEARCH_DIRS)
    if(EXISTS "${_dir}")
        list(APPEND _search_dirs "${_dir}")
    endif()
endforeach()

# file(GET_RUNTIME_DEPENDENCIES) always also searches the directory the
# executable lives in. DEST already holds DLLs copied by a previous target, and
# those collide with the same DLL in the search dirs:
#   "file Multiple conflicting paths found for libgcc_s_seh-1.dll"
# So copy the exe into a scratch directory first and analyse it there.
#
# The scratch directory MUST be unique per target: with "cmake --build -j"
# several targets run this script at the same time and a shared directory
# makes them delete each other's files (this used to break parallel builds).
get_filename_component(_exe_name "${EXE}" NAME)
set(_scan_dir "${DEST}/.runtime_scan/${_exe_name}")
file(REMOVE_RECURSE "${_scan_dir}")
file(MAKE_DIRECTORY "${_scan_dir}")
file(COPY "${EXE}" DESTINATION "${_scan_dir}")

file(GET_RUNTIME_DEPENDENCIES
    EXECUTABLES "${_scan_dir}/${_exe_name}"
    DIRECTORIES ${_search_dirs}
    RESOLVED_DEPENDENCIES_VAR _resolved
    UNRESOLVED_DEPENDENCIES_VAR _unresolved
)

file(REMOVE_RECURSE "${_scan_dir}")

foreach(_dll IN LISTS _resolved)
    # System DLLs (Kernel32, msvcrt, ...) must not be copied.
    if(NOT _dll MATCHES "[Ss]ystem32")
        file(COPY "${_dll}" DESTINATION "${DEST}")
    endif()
endforeach()
