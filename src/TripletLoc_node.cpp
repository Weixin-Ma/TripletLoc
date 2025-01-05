#include <ros/ros.h>
#include "./../include/TripletLoc.h"

int main(int argc, char **argv) {
  ros::init(argc, argv, "tripletloc");
  ros::NodeHandle nh("~");

  TripletLoc node(nh);

  node.run();
  return 0;
}

