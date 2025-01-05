cmake_minimum_required(VERSION 3.10)
project(tripletloc)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -pthread")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  tf
  rosbag
)

find_package(PkgConfig REQUIRED)
pkg_check_modules(YAMLCPP yaml-cpp REQUIRED)
if(YAMLCPP_VERSION VERSION_GREATER "0.5.0")
    add_definitions(-DHAVE_YAMLCPP_GT_0_5_0)
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(PCL REQUIRED)
include_directories(include ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# Parallel Maximum Clique (pmc)
configure_file(pmc/pmc.CMakeLists.txt.in pmc-download/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/pmc-download")
execute_process(COMMAND "${CMAKE_COMMAND}" --build .
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/pmc-download")
add_subdirectory("${CMAKE_BINARY_DIR}/pmc-src"   #CMAKE_BINARY_DIR =>
        "${CMAKE_BINARY_DIR}/pmc-build")


find_package(GTSAM REQUIRED QUIET)

catkin_package(
        INCLUDE_DIRS
          include
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
)

# add_executable(filter_pc_heliper src/preprocoss_filter_pc_heliper.cpp ) #for filtering point cloud to fit spvnas
# target_link_libraries(filter_pc_heliper ${catkin_LIBRARIES} ${PCL_LIBRARIES})

# add_executable(generate_maps src/TripletLoc_generate_maps.cpp)   #generate instance map and RSN map
# target_link_libraries(generate_maps ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${YAMLCPP_LIBRARIES})

# ############# baseline Triplet-Graph ("Triplet-Graph: Global Metric Localization Based on Semantic Triplet Graph for Autonomous Vehicles", IEEE RA-L, 2024.)
# add_executable(baseline_TG src/baseline_TripletGraph.cpp src/tripletgraph.cpp)
# target_link_libraries(baseline_TG  ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${YAMLCPP_LIBRARIES} ${CERES_LIBRARIES})

# ############# our method 
set(TL_src
  src/teaser/graph.cc
  src/teaser/registration.cc

  src/TripletLoc.cpp
  src/TripletLoc_node.cpp
)

add_executable(tripletloc_node ${TL_src})
target_link_libraries(tripletloc_node ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${YAMLCPP_LIBRARIES} pmc gtsam)
