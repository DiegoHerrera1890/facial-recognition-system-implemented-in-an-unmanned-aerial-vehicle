#!/usr/bin/env python
import rospy
import os
import rospkg
import csv
import re

rospack = rospkg.RosPack()
rospack.get_path('soka_drone')

Machine = []
Command = []
Command_Code = []
Command_Parameters = []
# extracting commands and respective HAIVE from CSV file
with open(rospack.get_path('soka_drone')+'/Scripts/test_2.csv','r') as File:
    reader = csv.DictReader(File, delimiter=',', quotechar='|')
    for row in reader:
        Command.append(row['Command'])

# extracting information from commands
for item in Command:  # /\d+\.?\d*/
    x, y, z = list(map(float, re.findall(r'\d+\.?\d*', item[0:])))
    #z = z*(-1)
    print("Z: ", z)
    Command_Parameters.append((x, y, z))   # storing serial number and respective parameters
