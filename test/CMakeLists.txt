add_executable( run_vo run_vo.cpp )
target_link_libraries( run_vo myslam )
target_link_libraries(run_vo /home/lucky/liukun/libjson_linux-gcc-5.4.0_libmt.so)