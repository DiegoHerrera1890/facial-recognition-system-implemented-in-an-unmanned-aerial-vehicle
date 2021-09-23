

*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* Gazebo Simulator *-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*
diego@diego-desktop:~/src/PX4-Autopilot$ sudo nosim=1 make px4_sitl_default gazebo   // optional: px4_sitl gazebo_iris_opt_flow
diego@diego-desktop:~$ roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557"
diego@subarashi:~$ rosrun soka_drone 1_Quadcopter_node.py          *First
diego@subarashi:~$ rosrun soka_drone 2_distributor_node.py         *Second
diego@subarashi:~$ rosrun face_detection 5_face_detection_node.py   *Third
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-

*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* Real World *-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-**-*-*-*-*-**-*-*-
diego@subarashi:~$ sudo chmod 666 /dev/ttyTHS2  (Terminal 1)
diego@subarashi:~$ roslaunch mavros px4.launch fcu_url:="/dev/ttyTHS2:921600" gcs_url:="udp://:14401@127.0.0.1:14550" (Terminal 1)
diego@subarashi:~$ roslaunch px4_realsense_bridge bridge.launch  (Terminal 2)
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-

*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* Codes to run *-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-**-*-*-*-*-**-*
diego@subarashi:~$ rosrun soka_drone 1_Quadcopter_node.py           *First            (Terminal 3)
diego@subarashi:~$ rosrun soka_drone 2_distributor_node.py          *Second           (Terminal 4)
diego@subarashi:~$ rosrun face_detection 5_face_detection_node.py   *Third            (Terminal 5)
diego@subarashi:~$ rosrun soka_drone 5_face_detection_node.py        *Four            still improving
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-



***************************************************** Running Gazebo and on a particular drone ******************************************
Do nor run these scripts unless you need to choose a different drone model and not the default drone.

diego@diego-ubuntu:~/PX4-Autopilot$ roslaunch gazebo_ros empty_world.launch world_name:=$(pwd)/Tools/sitl_gazebo/worlds/iris.world
diego@diego-ubuntu:~/PX4-Autopilot$ no_sim=1 make px4_sitl_default gazebo
diego@diego-ubuntu:~$ roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557"
diego@diego-ubuntu:~$ rostopic echo /gazebo/model_states
diego@diego-ubuntu:~$ rostopic info /gazebo/model_states 
diego@subarashi:~$ rosrun soka_drone 1_Quadcopter_node.py           *First
diego@subarashi:~$ rosrun soka_drone 2_distributor_node.py          *Second
diego@subarashi:~$ rosrun face_detection 5_face_detection_node.py   *Third
diego@subarashi:~$ rosrun soka_drone 5_face_detection_node.py 
diego@subarashi:~$ rosrun soka_drone 3_searching_node.py            *Four








---------------------------------- Only for testing ------------------------------------------------------------------------------------------------------------------

rostopic pub -r 10 /Face_recognition/Searching std_msgs/String "Searching"

rostopic pub -r 10 /setpoint_attitude/attitude geometry_msgs/PoseStamped '{pose: {pose: {x: 0.0, y: 0.0, z: 2.0}, orientation: {x: 0.0,y: 0.0,z: 0.0}}}'
header.seq header.stamp header.frame_id pose.position.x pose.position.y pose.position.z pose.orientation.x pose.orientation.y pose.orientation.z pose.orientation.w

rostopic pub -r 10 /cmd_vel geometry_msgs/Twist  '{linear:  {x: 0.1, y: 0.0, z: 0.0}, angular: {x: 0.0,y: 0.0,z: 0.0}}'



rostopic pub -r 100 /mavros/setpoint_raw/local mavros_msgs/PositionTarget "header:
  seq: 0
  stamp: {secs: 0, nsecs: 0}
  frame_id: ''
coordinate_frame: 8
type_mask: 1475
position: {x: 0.0, y: 0.0, z: 1.6}
velocity: {x: 0.0, y: 0.0, z: 0.0}
acceleration_or_force: {x: 0.0, y: 0.0, z: 0.0}

yaw: 0.0
yaw_rate: 0.5"


> yaw_rate 1.0"



rostopic pub -r 100 /mavros/setpoint_raw/local mavros_msgs/PositionTarget "header:
  seq: 0
  stamp: {secs: 0, nsecs: 0}
  frame_id: ''
coordinate_frame: 8
type_mask: 1475
position: {x: 0.0, y: 0.0, z: 5.0}
velocity: {x: 1.0, y: 0.0, z: 0.0}
acceleration_or_force: {x: 0.0, y: 0.0, z: 0.0}

yaw: 0.0
yaw_rate: 1.0"


