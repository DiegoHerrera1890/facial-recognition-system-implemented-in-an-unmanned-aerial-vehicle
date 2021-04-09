diego@subarashi:~/src/Firmware$ sudo nosim=1 make px4_sitl_default gazebo
roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557"
px4_sitl gazebo_iris_opt_flow
diego@subarashi:~$ roslaunch mavros px4.launch fcu_url:="/dev/ttyTHS2:921600" gcs_url:="udp://:14401@127.0.0.1:14550"

diego@subarashi:~$ rosrun soka_drone Main_node.py
diego@subarashi:~$ rosrun soka_drone Quadcopter_1.py
diego@subarashi:~$ rosrun soka_drone diego_test.py

roslaunch mavros