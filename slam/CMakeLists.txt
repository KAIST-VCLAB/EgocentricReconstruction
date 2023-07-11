add_subdirectory(openvslam)

add_subdirectory(pangolin_viewer)

# Build main program
# ----- Find dependencies -----
# popl
set(popl_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/3rd/popl/include)
set(CMAKE_CXX_STANDARD 11)

# ----- Build main program -----
add_executable(video_pose video_pose_main.cc)

# Set output directory for executables
set_target_properties(video_pose PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PROJECT_BINARY_DIR}"
                        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PROJECT_BINARY_DIR}"
                        RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${PROJECT_BINARY_DIR}"
                        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${PROJECT_BINARY_DIR}")

# PangolinViewer is used on a priority basis
# Set macro flag
target_compile_definitions(video_pose PRIVATE USE_PANGOLIN_VIEWER)
# Link viewer
target_link_libraries(video_pose PRIVATE pangolin_viewer)

# Link OpenVSLAM
target_link_libraries(video_pose
                        PRIVATE
                        ${PROJECT_NAME}
                        opencv_imgcodecs
                        opencv_videoio)

# include popl and spdlog headers
target_include_directories(video_pose
                            PRIVATE
                            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/3rd/popl/include>
                            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/3rd/spdlog/include>)
