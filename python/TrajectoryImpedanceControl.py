"""
Impedance controlled tracking of trajectories of type ReferenceTrajectory

By Linda van der Spaa, 2022
l.f.vanderspaa@tudelft.nl
"""

import numpy as np
import time
import pickle

import rospy
from ReferenceTrajectories import ReferenceTrajectory
from WardrobeScenTrajectories import stateActionTrajectorySet, goalTrajectorySet
from geometry_msgs.msg import WrenchStamped, PoseStamped
from std_msgs.msg import Float32MultiArray, Int16MultiArray, Int8, Bool
from sensor_msgs.msg import JointState
import quaternion as Q

#%%
class VariableImpedanceController:
  NMA = 50
  last_forces_ = np.zeros((NMA,3))
  last_torques_ = np.zeros((NMA,3))
  stiffness = np.array([400, 400, 400, 30, 30, 30, 0])

  minJoints = [-2.75, -1.6, -2.75, -2.9, -2.75, -0.0, -2.7]
  maxJoints = [ 2.75,  1.6,  2.75, -0.2,  2.75,  3.6,  2.7]

  def ee_pos_callback(self, data): 
    self.curr_pos = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
    curr_ori_arr = np.array([data.pose.orientation.w, data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z])
    if curr_ori_arr[0] < 0:
      curr_ori_arr *= -1
    self.curr_ori = Q.from_float_array(curr_ori_arr)

  def read_conf(self, data):
    self.curr_conf = np.array(data.position[:7])
    self.curr_cVel = np.array(data.velocity[:7])
      
  def read_force(self, data):
    self.last_forces_ = np.append([[data.wrench.force.x, data.wrench.force.y, data.wrench.force.z]], self.last_forces_[:-1], axis=0)
    self.force = np.mean(self.last_forces_, axis=0)

    self.last_torques_ = np.append([[data.wrench.torque.x, data.wrench.torque.y, data.wrench.torque.z]], self.last_torques_[:-1], axis=0)
    self.torque = np.mean(self.last_torques_, axis=0)

  def set_variable_stiffness(self, msg):
    self.varStiff = msg.data

  def set_trajectory(self, msg):
    self.activeControl = False
    if (trajKey := (msg.data[0], msg.data[1])) in stateActionTrajectorySet:
      newTrajectory = stateActionTrajectorySet[trajKey]
      newTrajectory.setStart(self.curr_pos, self.curr_ori)
      self.trajectory = newTrajectory
      if self.debugLevel >= 2:
        print(self.trajectory.xRefTraj)
      self.activeControl = True
      if self.debugLevel >= 1:
        print("Active control action {}".format(trajKey))

      self.newTraj = True

  def set_straight_trajectory(self, msg):
    self.activeControl = False
    if (trajKey := msg.data) in goalTrajectorySet:
      newTrajectory = goalTrajectorySet[trajKey]
      newTrajectory.setStart(self.curr_pos, self.curr_ori)
      self.trajectory = newTrajectory
      if self.debugLevel >= 2:
        print(self.trajectory.xRefTraj)
      self.activeControl = True
      if self.debugLevel >= 1:
        print("Active control to goal {}".format(trajKey))

      self.newTraj = True

  def set_passive(self, msg):
    if msg.data:
      self.setPassive()

  def set_debug_level(self, msg):
    self.debugLevel = msg.data

  def __init__(self, eeCtrlDOF=4, nullspaceControl=True) -> None:
    self.goal = PoseStamped()
    self.jointConfiguration = Float32MultiArray()
    self.stiff_des = Float32MultiArray()

    self.eeDOF = eeCtrlDOF
    self.debugLevel = 0
    self.varStiff = True
    self.nullspaceControl = nullspaceControl
    if self.nullspaceControl:
      with open('nullspace_gpgp0.pkl', 'rb') as f:
        self.gp_null = pickle.load(f)
        self.gp_null0 = pickle.load(f)
        # self.gp_nullX = pickle.load(f)
        # self.gp_nullY = pickle.load(f)
        self.std_nullMax = np.sqrt(self.gp_null.kernel_.get_params()['k1__k1__constant_value'])

      self.q_pub = rospy.Publisher('/equilibrium_configuration', Float32MultiArray, queue_size=0)
    # attractor publisher
    self.goal_pub = rospy.Publisher('/equilibrium_pose', PoseStamped, queue_size=0)
    self.stiff_pub = rospy.Publisher('/stiffness', Float32MultiArray, queue_size=0) #in this vector we can send [x, y, z, rotX, rotY, rotZ, ns] stiffness
    time.sleep(1)

    self.newTraj = False
    self.activeControl = False
    self.setPassive()

    rospy.Subscriber("/cartesian_pose", PoseStamped, self.ee_pos_callback)
    rospy.Subscriber("/joint_states", JointState, self.read_conf)
    rospy.Subscriber("/force_torque_ext", WrenchStamped , self.read_force)
    
    rospy.Subscriber('/DAVIcontroller/allow_disagreement', Bool, self.set_variable_stiffness)
    rospy.Subscriber('/DAVIcontroller/passive', Bool, self.set_passive)
    rospy.Subscriber('/DAVIcontroller/trajectory_indices', Int16MultiArray, self.set_trajectory)
    rospy.Subscriber('/DAVIcontroller/goal', Int8, self.set_straight_trajectory)
    rospy.Subscriber('/DAVIcontroller/debug', Int8, self.set_debug_level)
    time.sleep(1)

    self.setForceTransform(lambda f,t:f)

  def setForceTransform(self, transform:np.ndarray):
    self.forceAtHuman = transform

  def sendPoseToRobot(self, xSetpoint, qSetpoint):
    self.goal.pose.position.x = xSetpoint[0]
    self.goal.pose.position.y = xSetpoint[1]
    self.goal.pose.position.z = xSetpoint[2]

    self.goal.pose.orientation.x = qSetpoint.x
    self.goal.pose.orientation.y = qSetpoint.y
    self.goal.pose.orientation.z = qSetpoint.z
    self.goal.pose.orientation.w = qSetpoint.w

    if self.debugLevel >= 2:
      print("Goal:")
      print(self.goal.pose)
    self.goal_pub.publish(self.goal)

  def sendStiffness(self):
    if self.eeDOF < 4:
      stiff_ang = 25.
    self.stiff_des.data = self.stiffness.astype(np.float32)
    if self.debugLevel >= 2:
      print("Stiffness:")
      print(self.stiff_des)
    self.stiff_pub.publish(self.stiff_des)

  def run(self):
    xMargin = 1e-3        # acceptable goal error [m]
    qMargin = np.pi/180   # acceptable goal error [rad]
    xErr = 0.05           # trajectory error recognized as gR =/ gH [m]
    angErr = np.pi/4      # angular error recognized as gR =/ gH [rad]
    fErr = 5.             # intraction force recognized as gR =/ gH [N]
    tErr = 2.5            # intraction torque recognized as gR =/ gH [N]
    K_lin_max=600
    K_lin_min=250
    K_ang_max=15
    K_ang_min=10
    K_null_max=40
    K_null_min=5
    T_reduce_stiff = .5   # time of human resistance to drop stiffness [s]

    def isAtGoal(relPrecision:float=1.):
      diffX, diffOri = self.trajectory.getDiff2Goal(self.curr_pos, self.curr_ori)
      return diffX < xMargin*relPrecision and diffQ < qMargin*relPrecision

    def getTrajectoryStiffness(K_null:float=K_null_max):
      if not self.nullspaceControl:
        return K_lin_max, K_ang_max, 0.

      diffX_s, diffOri_s, diffX_g, diffOri_g = self.trajectory.getDiff2StartGoal(self.curr_pos, self.curr_ori)

      d_lin_min = 0.01;     d_lin_max = 0.1
      d_lin = min(diffX_g,diffX_s)
      alpha_lin = 1 if d_lin <= d_lin_min else \
                  0 if d_lin >= d_lin_max else \
                  (d_lin_max - d_lin)/(d_lin_max - d_lin_min)

      d_ang_min = np.pi/36; d_ang_max = np.pi/9
      d_ang = min(diffOri_g, diffOri_s)
      alpha_ang = 1 if d_ang <= d_ang_min else \
                  0 if d_ang >= d_ang_max else \
                  (d_ang_max - d_ang)/(d_ang_max - d_ang_min)

      alpha = min(alpha_lin, alpha_ang)
      K_lin = K_lin_min + alpha*(K_lin_max - K_lin_min)
      K_ang = K_ang_min + alpha*(K_ang_max - K_ang_min)

      if diffX_g < d_lin_min*2 and diffOri_g < d_ang_min*2:
        K_null = max(K_null_min, K_null - dt/.5*(K_null_max-K_null_min))
      else:
        K_null = min(K_null_max, K_null + dt/.5*(K_null_max-K_null_min))
      return K_lin, K_ang, K_null

    # control loop
    dt = 1/30   # loop_time = 1/control_frequency
    r = rospy.Rate(1/dt)

    stiff_null = K_null_min
    interactionStiffness=1.
    noMotionIter = 0
    old_pose = [self.curr_pos, self.curr_ori]


    while not rospy.is_shutdown():
      if self.activeControl:
        # compute setpoint
        xSetpoint, qSetpoint = self.trajectory.getSetpoint(dt, self.curr_pos, self.curr_ori)

        if self.debugLevel >= 1:
          if (err:=np.linalg.norm(self.force)) > fErr:
            print('High force detected: {0} N'.format(err))
          if (err:=np.linalg.norm(self.torque)) > tErr:
            print('High torque detected: {0} Nm'.format(err))
          if (err:=np.linalg.norm(xSetpoint-self.curr_pos)) > xErr:
            print('Large distance error: {0} m'.format(err))
          if (err:=Q.rotation_intrinsic_distance(qSetpoint, self.curr_ori)) > angErr:
            print('Large angular error: {0} rad'.format(err))

        stiff_lin, stiff_ang, stiff_null = getTrajectoryStiffness(stiff_null)
        if self.varStiff:
          # check if stiffness needs to be reduced
          # reduce_stiff = np.linalg.norm(self.force) > fErr or np.linalg.norm(xSetpoint-self.curr_pos) > xErr or np.linalg.norm(self.torque) > tErr or Q.rotation_intrinsic_distance(qSetpoint, self.curr_ori) > angErr
          if np.linalg.norm(self.force) > fErr or np.linalg.norm(self.torque) > tErr:
            interactionStiffness = max(interactionStiffness-dt/T_reduce_stiff, 0)
          elif interactionStiffness > 0:
            interactionStiffness = min(interactionStiffness+dt/T_reduce_stiff, 1)

          stiff_lin *= interactionStiffness
          stiff_ang *= interactionStiffness
          stiff_null *= interactionStiffness
        
        # Correct stiffness to avoid sudden increase
        if stiff_lin > self.stiffness[0] + (maxDeltaK := K_lin_max*dt/T_reduce_stiff):
          stiff_lin = self.stiffness[0] + maxDeltaK
        if stiff_ang > self.stiffness[5] + (maxDeltaKappa := K_ang_max*dt/T_reduce_stiff):
          stiff_ang = self.stiffness[5] + maxDeltaKappa
        if stiff_null > self.stiffness[6] + (maxDeltaNullK := K_null_max*dt/T_reduce_stiff):
          stiff_null = self.stiffness[6] + maxDeltaNullK

        # send goal position, angle
        self.sendPoseToRobot(xSetpoint, qSetpoint)

        if self.nullspaceControl and self.activeControl:
          goalPoseArray = np.hstack([xSetpoint, Q.as_float_array(qSetpoint)])
          configuration = self.gp_null.predict([goalPoseArray]) + self.gp_null0
          configuration = np.minimum(np.maximum(configuration, self.minJoints), self.maxJoints)

          if np.max(np.abs(configuration - self.curr_conf)) > 0.5:
            stiff_null = min(stiff_null, 2.)

          self.jointConfiguration.data = configuration[0].astype(np.float32)
          if self.debugLevel >= 2:
            print("Joint configuration:")
            print(self.jointConfiguration.data)
          self.q_pub.publish(self.jointConfiguration)

        self.stiffness = np.array([stiff_lin, stiff_lin, stiff_lin, 25., 25., stiff_ang, stiff_null])
        self.sendStiffness()

        #use rate of ROS
        r.sleep()

        # exit when goal is reached
        if isAtGoal():
          interactionStiffness = max(interactionStiffness-dt/T_reduce_stiff, 0)
          self.setPassive()

        # exit when stiffness is dropped
        if self.stiffness[0] < 1e-9 and (self.eeDOF < 4 or self.stiffness[5] < 1e-9):
          if not self.newTraj:
            if self.debugLevel >= 1:
              print("Active control switched off because of zero stiffness")
            self.setPassive()
          else:
            if self.debugLevel >= 1:
              print("Starting new trajectory")
            interactionStiffness = 1.
            self.newTraj = False

        # exit when standing still
        if np.linalg.norm(self.curr_pos - old_pose[0]) < xMargin and Q.rotation_intrinsic_distance(self.curr_ori, old_pose[1]) < qMargin:
          noMotionIter += 1
          if noMotionIter > 0.5/dt: # 0.5 sec
            self.setPassive()
        else:
          noMotionIter = 0
        old_pose = [self.curr_pos, self.curr_ori]

    return 0

  def setPassive(self):
    self.activeControl = False
    if self.debugLevel >= 1:
      print("Active control switched off in passive mode")

    self.stiffness = np.array([0.0, 0.0, 0.0, 25.0, 25.0, 25.0 if self.eeDOF<4 else 0.0, 0.0])
    self.sendStiffness()
    return 0

#%%
# Main function
if __name__ == '__main__':
  rospy.init_node('impedance_controller')

  try:
    controller = VariableImpedanceController()
    controller.run()
  except rospy.ROSInterruptException: pass