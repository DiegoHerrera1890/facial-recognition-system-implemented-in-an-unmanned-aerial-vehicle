#include <ros/ros.h>
#include <std_msgs/Int32MultiArray.h>
#include <eigen_conversions/eigen_msg.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Quaternion.h>
#include <string>
#include <venom_offb/Navigator.h>
#include <venom_perception/Zed.h>
#include "util.h"


#if INCLUDE_NAVIGATOR == 1
venom::Navigator* nav;
#endif

void exit_handler(int s) {                                                      
  ROS_WARN("Force quitting...\n");
  #if INCLUDE_NAVIGATOR == 1
  nav->Land();
  delete nav;
  #endif
  exit(1);
}

int main(int argc, char** argv) {

  ros::init(argc, argv, "visual_servo", ros::init_options::NoSigintHandler);

  signal(SIGINT, exit_handler);

  venom::Zed zed;
  zed.Enable(venom::PerceptionType::ODOM);

  #if INCLUDE_NAVIGATOR == 1
  nav = new venom::Navigator();
  nav->TakeOff(1.0);
  #endif

  while (ros::ok()) {
    ros::spinOnce();
  }

  return 0;
}