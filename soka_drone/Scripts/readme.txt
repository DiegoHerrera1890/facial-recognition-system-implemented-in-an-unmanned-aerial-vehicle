

*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* Gazebo Simulator *-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*
diego@diego-desktop:~/src/PX4-Autopilot$ sudo nosim=1 make px4_sitl_default gazebo   // optional: px4_sitl gazebo_iris_opt_flow
diego@diego-desktop:~$ roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557"
diego@subarashi:~$ rosrun soka_drone 1_Quadcopter_node.py          *First
diego@subarashi:~$ rosrun soka_drone 2_distributor_node.py         *Second
diego@subarashi:~$ rosrun face_detection 5_face_detection_node.py   *Third
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-

*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* Real World ZED Camera *-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-**-*-*-*-*-**-*-*-
sudo date MMDDHHMMYYYY.SS
diego@subarashi:~$ roslaunch zed_wrapper zedm.launch (Terminal 2)
diego@subarashi:~$ rosrun venom_offb vision_pose_converter (Terminal 3)

diego@subarashi:~$ sudo chmod 666 /dev/ttyTHS2  (Terminal 1)
diego@subarashi:~$ roslaunch mavros px4.launch fcu_url:="/dev/ttyTHS2:921600" gcs_url:="udp://:14401@127.0.0.1:14550" (Terminal 1)
diego@subarashi:~$ rosrun soka_drone offb_node.py

rostopic echo /zedm/zed_node/odom
rostopic echo /mavros/local_position/pose

diego@subarashi:~$ rosrun soka_drone 1_Quadcopter_node.py           *First            (Terminal 4)
diego@subarashi:~$ rosrun soka_drone takeoff_landing.py                               (Terminal 5)

rostopic pub /Face_recognition/model_ready std_msgs/String ready                      (Terminal 6)
rostopic pub /Face_recognition/landing/kill_searching std_msgs/String landing 
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*_*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-**-*-*-*-*-**-*




*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* Real World realsense T265 *-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-**-*-*-*-*-**-*-*-
# sudo date MMDDHHMMYYYY.SS
xonapa@drone:~$ sudo chmod 666 /dev/ttyTHS2  (Terminal 1)
xonapa@drone:~$ roslaunch mavros px4.launch fcu_url:="/dev/ttyTHS2:3000000" gcs_url:="udp://:14401@127.0.0.1:14550" (Terminal 1)
xonapa@drone:~$ roslaunch px4_realsense_bridge bridge.launch  (Terminal 2)
xonapa@drone:~$ rostopic echo /camera/odom/sample
xonapa@drone:~$ rostopic echo /mavros/local_position/pose
xonapa@drone:~$ rosrun image_view video_recorder image:=/camera/fisheye1/image_raw _image_transport:="compressed" _filename:="/media/jetson/B2E4-69EA3/videos/4/fisheye_1.avi" _fps:="15" _codec:="I420"
xonapa@drone:~$ rosrun image_view video_recorder image:=/camera/fisheye2/image_raw _image_transport:="compressed" _filename:="/media/jetson/B2E4-69EA3/videos/4/fisheye_2.avi" _fps:="15" _codec:="I420"


xonapa@drone:~$ rostopic pub /Face_recognition/model_ready std_msgs/String ready  (Terminal 2)
rostopic pub /Face_recognition/Searching std_msgs/String Searching

*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# diego@subarashi:~$ roslaunch ros_deep_learning detectnet.ros1.launch input:=v4l2:///dev/video1 output:=display://0
sudo chmod 666 /sys/class/gpio/gpio388/value
sudo chmod 666 /sys/class/gpio/gpio298/value
sudo chmod 666 /sys/class/gpio/gpio480/value
sudo chmod 666 /sys/class/gpio/gpio486/value

*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* Codes to run *-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-**-*-*-*-*-**-*
diego@subarashi:~$ rosrun soka_drone 1_Quadcopter_node.py           *First            (Terminal 3)
diego@subarashi:~$ rosrun soka_drone 2_distributor_node.py          *Second           (Terminal 4)
diego@subarashi:~$ rosrun face_detection 5_face_detection_node.py   *Third            (Terminal 5)
diego@subarashi:~$ rosrun soka_drone 5_face_detection_node.py        *Four            still improving


diego@subarashi:~$ rosrun soka_drone Coordinates_reader.py
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-



***************************************************** Running Gazebo and on a particular drone ***********************************
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




rostopic pub -r 100 /mavros/setpoint_raw/local mavros_msgs/PositionTarget "header:
  seq: 0
  stamp: {secs: 0, nsecs: 0}
  frame_id: ''
coordinate_frame: 8
type_mask: 1475
yaw: 0.0
yaw_rate: 1.0"





rostopic pub -r 10 /mavros/setpoint_attitude/cmd_vel geometry_msgs/TwistStamped "header:
  seq: 0
  stamp: {secs: 0, nsecs: 0}
  frame_id: ''
twist:
  linear: {x: 0.0, y: 0.0, z: 1.6}
  angular: {x: 0.0, y: 0.0, z: 0.5}"





rostopic pub /Face_recognition/Searching std_msgs/String "Searching"
