# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

if(EXISTS "G:/test/pp-test/ypc_hrcoil/dabaodll/cpp_infer_tps_cpu/third-party/extern_autolog-subbuild/autolog/src/extern_autolog-populate-stamp/extern_autolog-populate-gitclone-lastrun.txt" AND EXISTS "G:/test/pp-test/ypc_hrcoil/dabaodll/cpp_infer_tps_cpu/third-party/extern_autolog-subbuild/autolog/src/extern_autolog-populate-stamp/extern_autolog-populate-gitinfo.txt" AND
  "G:/test/pp-test/ypc_hrcoil/dabaodll/cpp_infer_tps_cpu/third-party/extern_autolog-subbuild/autolog/src/extern_autolog-populate-stamp/extern_autolog-populate-gitclone-lastrun.txt" IS_NEWER_THAN "G:/test/pp-test/ypc_hrcoil/dabaodll/cpp_infer_tps_cpu/third-party/extern_autolog-subbuild/autolog/src/extern_autolog-populate-stamp/extern_autolog-populate-gitinfo.txt")
  message(STATUS
    "Avoiding repeated git clone, stamp file is up to date: "
    "'G:/test/pp-test/ypc_hrcoil/dabaodll/cpp_infer_tps_cpu/third-party/extern_autolog-subbuild/autolog/src/extern_autolog-populate-stamp/extern_autolog-populate-gitclone-lastrun.txt'"
  )
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -rf "G:/test/pp-test/ypc_hrcoil/dabaodll/cpp_infer_tps_cpu/third-party/extern_autolog-src"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: 'G:/test/pp-test/ypc_hrcoil/dabaodll/cpp_infer_tps_cpu/third-party/extern_autolog-src'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "D:/Program Files/Git/cmd/git.exe"
            clone --no-checkout --config "advice.detachedHead=false" "https://github.com/LDOUBLEV/AutoLog.git" "extern_autolog-src"
    WORKING_DIRECTORY "G:/test/pp-test/ypc_hrcoil/dabaodll/cpp_infer_tps_cpu/third-party"
    RESULT_VARIABLE error_code
  )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once: ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/LDOUBLEV/AutoLog.git'")
endif()

execute_process(
  COMMAND "D:/Program Files/Git/cmd/git.exe"
          checkout "main" --
  WORKING_DIRECTORY "G:/test/pp-test/ypc_hrcoil/dabaodll/cpp_infer_tps_cpu/third-party/extern_autolog-src"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: 'main'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "D:/Program Files/Git/cmd/git.exe" 
            submodule update --recursive --init 
    WORKING_DIRECTORY "G:/test/pp-test/ypc_hrcoil/dabaodll/cpp_infer_tps_cpu/third-party/extern_autolog-src"
    RESULT_VARIABLE error_code
  )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: 'G:/test/pp-test/ypc_hrcoil/dabaodll/cpp_infer_tps_cpu/third-party/extern_autolog-src'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy "G:/test/pp-test/ypc_hrcoil/dabaodll/cpp_infer_tps_cpu/third-party/extern_autolog-subbuild/autolog/src/extern_autolog-populate-stamp/extern_autolog-populate-gitinfo.txt" "G:/test/pp-test/ypc_hrcoil/dabaodll/cpp_infer_tps_cpu/third-party/extern_autolog-subbuild/autolog/src/extern_autolog-populate-stamp/extern_autolog-populate-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: 'G:/test/pp-test/ypc_hrcoil/dabaodll/cpp_infer_tps_cpu/third-party/extern_autolog-subbuild/autolog/src/extern_autolog-populate-stamp/extern_autolog-populate-gitclone-lastrun.txt'")
endif()
