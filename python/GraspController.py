"""
Controller to handle grasping of object

Execute on top of Franka Human Friendly Controllers

By Linda van der Spaa, 2022
l.f.vanderspaa@tudelft.nl
"""

import rospy, time
from std_msgs.msg import Float32, Bool, Int8
from sensor_msgs.msg import JointState
from franka_gripper.msg import GraspActionGoal, StopActionGoal, HomingActionGoal

class GraspController:
  def gripper_pos_callback(self, data):
    pos = data.position[0]
    # Object of a width between 6 and 10 mm
    self.curr_state = 0 if pos > 0.01/2 else 1 if pos > 0.004/2 else -1
    self.state_pub.publish(self.curr_state)

  def open(self, dOpen=0.02): # open distance [m]
    if self.last_grip_command == False:
      # exit without acting if previous command sent was the same
      return 0 if self.curr_state == 0 else -1

    self.grip_command.goal.epsilon.inner = 0.0
    self.grip_command.goal.epsilon.outer = 0.0
    self.grip_command.goal.force = 0.
    self.grip_command.goal.width = dOpen
    self.gripper_pub.publish(self.grip_command)
    self.last_grip_command = False

    r = rospy.Rate(20)
    while not rospy.is_shutdown():
      if self.curr_state == 0:
        return 0  # exit flag
      r.sleep()

  def close(self, dClose=0.0): # close distance [m]
    if self.last_grip_command:
      # exit without acting if previous command sent was the same
      return 0 if self.curr_state else -1

    self.grip_command.goal.epsilon.inner = 0.1
    self.grip_command.goal.epsilon.outer = 0.1
    self.grip_command.goal.force = 1000.
    self.grip_command.goal.width = dClose
    self.gripper_pub.publish(self.grip_command)
    self.last_grip_command = True

    r = rospy.Rate(20)
    while not rospy.is_shutdown():
      if self.curr_state:
        return 0  # exit flag
      r.sleep()

  def grasp_callback(self, grasp):
    self.goal_state = 1 if grasp.data else 0

  def homing_callback(self, _):
    self.goal_state = 0
    self.last_grip_command = None
    self.stop_pub.publish(self.stop)
    self.homing_pub.publish(self.home)
    # self.open()

  def __init__(self) -> None:
    self.stop = StopActionGoal()
    self.home = HomingActionGoal()
    self.grip_command = GraspActionGoal()
    self.grip_command.goal.speed = 1.
    self.goal_state = 0
    self.last_grip_command = None

    self.state_pub = rospy.Publisher('/grasp_controller/state', Int8, queue_size=0)
    self.gripper_pub = rospy.Publisher("franka_gripper/grasp/goal", GraspActionGoal, queue_size=0)
    self.stop_pub = rospy.Publisher("/franka_gripper/stop/goal", StopActionGoal, queue_size=0)
    self.homing_pub = rospy.Publisher("/franka_gripper/homing/goal", HomingActionGoal, queue_size=0)
    time.sleep(1)

    rospy.Subscriber("/franka_gripper/joint_states", JointState, self.gripper_pos_callback)
    rospy.Subscriber("/grasp_controller/grasp", Bool, self.grasp_callback)
    rospy.Subscriber("/grasp_controller/home_gripper", Bool, self.homing_callback)
    time.sleep(1)

  def run(self) -> None:
    r = rospy.Rate(20)
    while not rospy.is_shutdown():
      if self.goal_state == 0 and self.curr_state != 0 and self.last_grip_command:
        self.open()
      
      elif self.goal_state == 1 and self.curr_state != self.goal_state:
        if self.curr_state < 0 and self.last_grip_command:
          print("Warning: lost the object, trying to regrasp")
          self.open()
        elif self.curr_state == 0 and not self.last_grip_command:
          self.close()

      r.sleep()


# Main function
if __name__ == '__main__':
  rospy.init_node('grasp_controller', anonymous=True)

  try:
    graspController = GraspController()
    graspController.run()
  except rospy.ROSInterruptException: pass
