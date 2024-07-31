# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "G:/test/pp-test/ypc_hrcoil/dabaodll/cpp_infer_tps_cpu/third-party/extern_autolog-src"
  "G:/test/pp-test/ypc_hrcoil/dabaodll/cpp_infer_tps_cpu/third-party/extern_autolog-build"
  "G:/test/pp-test/ypc_hrcoil/dabaodll/cpp_infer_tps_cpu/third-party/extern_autolog-subbuild/autolog"
  "G:/test/pp-test/ypc_hrcoil/dabaodll/cpp_infer_tps_cpu/third-party/extern_autolog-subbuild/autolog/tmp"
  "G:/test/pp-test/ypc_hrcoil/dabaodll/cpp_infer_tps_cpu/third-party/extern_autolog-subbuild/autolog/src/extern_autolog-populate-stamp"
  "G:/test/pp-test/ypc_hrcoil/dabaodll/cpp_infer_tps_cpu/third-party/extern_autolog-subbuild/autolog/src"
  "G:/test/pp-test/ypc_hrcoil/dabaodll/cpp_infer_tps_cpu/third-party/extern_autolog-subbuild/autolog/src/extern_autolog-populate-stamp"
)

set(configSubDirs Debug)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "G:/test/pp-test/ypc_hrcoil/dabaodll/cpp_infer_tps_cpu/third-party/extern_autolog-subbuild/autolog/src/extern_autolog-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "G:/test/pp-test/ypc_hrcoil/dabaodll/cpp_infer_tps_cpu/third-party/extern_autolog-subbuild/autolog/src/extern_autolog-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
