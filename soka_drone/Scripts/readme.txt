*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* Gazebo Simulator *-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*
diego@diego-desktop:~/src/PX4-Autopilot$ sudo nosim=1 make px4_sitl_default gazebo   // optional: px4_sitl gazebo_iris_opt_flow
diego@diego-desktop:~$ roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557"
diego@subarashi:~$ rosrun soka_drone 1_Quadcopter_node.py          *First
diego@subarashi:~$ rosrun soka_drone 2_distributor_node.py         *Second
diego@subarashi:~$ rosrun face_detection 5_face_detection_node.py   *Third
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-

*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* Real World *-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-**-*-*-*-*-**-*-*-
diego@subarashi:~$ sudo chmod 666 /dev/ttyTHS2
diego@subarashi:~$ roslaunch mavros px4.launch fcu_url:="/dev/ttyTHS2:921600" gcs_url:="udp://:14401@127.0.0.1:14550"
diego@subarashi:~$ roslaunch px4_realsense_bridge bridge.launch
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-

*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* Codes to run *-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-**-*-*-*-*-**-*
diego@subarashi:~$ rosrun soka_drone 1_Quadcopter_node.py           *First
diego@subarashi:~$ rosrun soka_drone 2_distributor_node.py          *Second
diego@subarashi:~$ rosrun face_detection 5_face_detection_node.py   *Third
diego@subarashi:~$ rosrun soka_drone 3_searching_node.py            *Four
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-


