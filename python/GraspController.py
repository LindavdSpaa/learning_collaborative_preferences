"""
Controller to handle grasping of object

Execute on top of Franka Human Friendly Controllers, after
> rosrun franka_gripper franka_gripper_online

By Linda van der Spaa, 2022
l.f.vanderspaa@tudelft.nl
"""

import rospy, time
from std_msgs.msg import Float32, Bool, Int8
from sensor_msgs.msg import JointState

class GraspController:
  def gripper_pos_callback(self, data):
    pos = data.position[0]
    # Object of a width between 2 and 20 mm
    self.curr_state = 0 if pos > 0.01 else 1 if pos > 0.001 else -1
    self.state_pub.publish(self.curr_state)

  def open(self, dOpen=0.1): # open distance [m]
    self.gripper_pub.publish(dOpen)

    while True:
      if self.curr_state == 0:
        time.sleep(0.5)
        return 0  # exit flag
      time.sleep(0.05)

  def close(self, dClose=0.0): # close distance [m]
    self.gripper_pub.publish(dClose)

    while True:
      if self.curr_state:
        time.sleep(0.2)
        return 0  # exit flag
      time.sleep(0.05)

  def grasp_callback(self, grasp):
    self.goal_state = 1 if grasp.data else 0

  def __init__(self) -> None:
    self.goal_state = 0
    self.gripper_pub = rospy.Publisher('/gripper_online', Float32, queue_size=0)
    self.state_pub = rospy.Publisher('/grasp_controller/state', Int8, queue_size=0)

    rospy.Subscriber("/franka_gripper/joint_states", JointState, self.gripper_pos_callback)
    rospy.Subscriber("/grasp_controller/grasp", Bool, self.grasp_callback)
    time.sleep(1)

  def run(self) -> None:
    r = rospy.Rate(1)
    while not rospy.is_shutdown():
      if self.goal_state == 0 and self.curr_state != 0:
        self.open()
      
      elif self.goal_state == 1 and self.curr_state != self.goal_state:
        if self.curr_state < 0:
          print("Warning: lost the object, trying to regrasp")
          self.open()
        self.close()

      r.sleep()


# Main function
if __name__ == '__main__':
  rospy.init_node('grasp_controller', anonymous=True)

  try:
    graspController = GraspController()
    graspController.run()
  except rospy.ROSInterruptException: pass