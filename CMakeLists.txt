#------------------------------------------------------------------------------
# Test face alignment algorithm
#------------------------------------------------------------------------------
project(bobetocalo_prl19)
cmake_minimum_required(VERSION 3.2)

include(${CMAKE_CURRENT_LIST_DIR}/../../CMakeLists.txt)

message("test\n")
#-- Find include files
list(APPEND face_alignment_bobetocalo_prl19_include
  ${faces_framework_include}
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/bazel-genfiles
   
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/tensorflow/contrib/makefile/downloads/eigen
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/tensorflow/contrib/makefile/downloads/absl
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/tensorflow/contrib/makefile/downloads/nsync/public
  ${CMAKE_CURRENT_LIST_DIR}/include
)

set(face_alignment_bobetocalo_prl19_src
  ${faces_framework_src}
  ${CMAKE_CURRENT_LIST_DIR}/src/FaceAlignmentPrl19.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/FaceDetectorOpencv.cpp
)

set(face_alignment_bobetocalo_prl19_test
        ${CMAKE_CURRENT_LIST_DIR}/test/face_alignment_bobetocalo_prl19_test.cpp
)

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_CURRENT_LIST_DIR}/tensorflow/bazel-bin/tensorflow/")

set(face_alignment_bobetocalo_prl19_libs
  ${faces_framework_libs}
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/bazel-bin/tensorflow/libtensorflow_cc.so
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/bazel-bin/tensorflow/libtensorflow_framework.so.1
  dl
  pthread
)

message("${face_alignment_bobetocalo_prl19_libs}")
#-- Setup CMake to run tests
enable_testing()

include_directories(${face_alignment_bobetocalo_prl19_include})

foreach(test ${face_alignment_bobetocalo_prl19_test})
  get_filename_component(test_name ${test} NAME_WE)
  add_executable(${test_name}
    ${face_alignment_bobetocalo_prl19_src}
    ${test}
  )
  target_link_libraries(${test_name} ${face_alignment_bobetocalo_prl19_libs})
  add_test(NAME ${test_name} WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR} COMMAND ${test_name} --database 300w_public)
endforeach()
