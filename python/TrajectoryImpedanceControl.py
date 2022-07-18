"""
Impedance controlled tracking of piece-wise linear trajectories,
online modulating the attractor distance following (Gams et al., 2009)

By Linda van der Spaa, 2022
l.f.vanderspaa@tudelft.nl
"""

import numpy as np
import time
import pickle

from numpy.core.fromnumeric import mean, prod
import rospy
from typing import Tuple
from geometry_msgs.msg import WrenchStamped, PoseStamped
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
import quaternion as Q

class ReferenceTrajectory:
  alpha_err_x = 10.
  alpha_err_q = 50.

  v = 0.3               # desired linear velocity [m/s]
  w = np.pi/2           # desired angular velocity [rad/s]

  tRefTraj = None
  xRefTraj = None
  qRefTraj = None

  def __init__(self, xyzWaypoints:np.array, oriWaypoints:np.array,
               obj2robotTransform:Tuple[np.array,Q.quaternion]=lambda p,q:(p,q),
               robot2objTransform:Tuple[np.array,Q.quaternion]=lambda p,q:(p,q)):
    tRefTraj = np.array([0])
    xRefTraj = xyzWaypoints
    qRefTraj = np.array([Q.from_float_array(o) for o in oriWaypoints])

    if (lenX := len(xRefTraj)) != (lenQ := len(qRefTraj)):
      raise Exception("Trajectory positions and orientations mut have same length. Found: len(pos) = {}, len(ori) = {}".format(lenX, lenQ))

    for i in range(1,lenX):
      xDist = np.linalg.norm(xRefTraj[i-1]-xRefTraj[i])
      angDist = Q.rotation_intrinsic_distance(qRefTraj[i-1], qRefTraj[i])
      tRefTraj = np.append(tRefTraj, [tRefTraj[-1] + max(xDist/self.v, angDist/self.w)])

    self.tRefTraj = tRefTraj
    self.xRefTraj = xRefTraj
    self.qRefTraj = qRefTraj

    self.xGoal = xRefTraj[-1]
    self.qGoal = qRefTraj[-1]

    self.transform = obj2robotTransform
    self.invTransform = robot2objTransform

  def initialize(self, xRobot:np.array, qRobot:Q.quaternion):
    x0, q0 = self.invTransform(xRobot, qRobot)
    xDist = np.linalg.norm(x0-self.xRefTraj[0])
    angDist = Q.rotation_intrinsic_distance(q0, self.qRefTraj[0])
    dt = max(xDist/self.v, angDist/self.w)

    self.tRefTraj = np.append([0], dt+self.tRefTraj)
    self.xRefTraj = np.append([x0], self.xRefTraj, axis=0)
    self.qRefTraj = np.append([q0], self.qRefTraj, axis=0)

    self.tRef = 0
    self.xRef = x0
    self.qRef = q0

  def getRobotGoal(self):
    return self.transform(self.xGoal, self.qGoal)

  def getDiff2Goal(self, xyzRobot, qRobot):
    xObj, qObj = self.invTransform(xyzRobot, qRobot)
    return np.linalg.norm(xObj-self.xGoal), Q.rotation_intrinsic_distance(qObj,self.qGoal)

  def setVelocities(self, v:float, w:float):
    self.v = v
    self.w = w

#%%
  def getSetpoint(self, dt:float, xRobot:np.array, qRobot:Q.quaternion):
    x, q = self.invTransform(xRobot, qRobot)
    correctionPQ = [1+self.alpha_err_x*np.linalg.norm(x-self.xRef),
                    1+self.alpha_err_q*Q.rotation_intrinsic_distance(q, self.qRef)]
    self.tRef += dt/(prod(correctionPQ)/mean(correctionPQ))
    if self.tRef < self.tRefTraj[-1]:
      self.xRef = np.array([np.interp(self.tRef, self.tRefTraj, self.xRefTraj[:,j]) for j in range(3)])

      tNextIdx = np.where(self.tRefTraj > self.tRef)[0][0]
      tPrevIdx = tNextIdx-1
      self.qRef = Q.slerp(self.qRefTraj[tPrevIdx], self.qRefTraj[tNextIdx], self.tRefTraj[tPrevIdx], self.tRefTraj[tNextIdx], self.tRef)
    else:
      self.xRef = self.xRefTraj[-1]
      self.qRef = self.qRefTraj[-1]
    return self.transform(self.xRef, self.qRef)

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

    if self.nullspaceControl and not self.activeControl:
      # t = time.time()
      goalPoseArray = np.hstack([self.curr_pos, curr_ori_arr])
      configuration, std_null = self.gp_null.predict([goalPoseArray], return_std=True)
      configuration += self.gp_null0
      configuration = np.minimum(np.maximum(configuration, self.minJoints), self.maxJoints)
      # Alternative prediction
      # KxX = self.gp_null.kernel_(goalPoseArray, self.gp_nullX)[0]
      # configuration = self.gp_null0 + KxX.dot(self.gp_nullY)/np.sum(KxX)

      jointConfiguration = Float32MultiArray()
      jointConfiguration.data = configuration[0].astype(np.float32)
      self.q_pub.publish(jointConfiguration)

      stiff_null = 20*np.max([0, 0.5-std_null[0,0]/self.std_nullMax])
      # stiff_null = 10*(1-std_null[0]/self.std_nullMax)
      self.stiffness = np.array([0.0, 0.0, 0.0, 30.0, 30.0, 0.0, stiff_null])
      stiff_des = Float32MultiArray()
      stiff_des.data = self.stiffness.astype(np.float32)
      self.stiff_pub.publish(stiff_des)
      # print("Computation time: {} s".format(time.time()-t))

  def read_conf(self, data):
    self.curr_conf = np.array(data.position[:7])
    self.curr_cVel = np.array(data.velocity[:7])
      
  def read_force(self, data):
    self.last_forces_ = np.append([[data.wrench.force.x, data.wrench.force.y, data.wrench.force.z]], self.last_forces_[:-1], axis=0)
    self.force = np.mean(self.last_forces_, axis=0)

    self.last_torques_ = np.append([[data.wrench.torque.x, data.wrench.torque.y, data.wrench.torque.z]], self.last_torques_[:-1], axis=0)
    self.torque = np.mean(self.last_torques_, axis=0)

  def __init__(self, eeCtrlDOF=4, nullspaceControl=True) -> None:
    self.eeDOF = eeCtrlDOF
    self.nullspaceControl = nullspaceControl
    if self.nullspaceControl:
      with open('nullspace_gpgp0.pkl', 'rb') as f:
        self.gp_null = pickle.load(f)
        self.gp_null0 = pickle.load(f)
        # self.gp_nullX = pickle.load(f)
        # self.gp_nullY = pickle.load(f)
        self.std_nullMax = np.sqrt(self.gp_null.kernel_.get_params()['k1__k1__constant_value'])

      self.q_pub = rospy.Publisher('/equilibrium_configuration', Float32MultiArray, queue_size=0)
    self.goal_pub = rospy.Publisher('/equilibrium_pose', PoseStamped, queue_size=0)
    self.stiff_pub = rospy.Publisher('/stiffness', Float32MultiArray, queue_size=0) #in this vector we can send [x, y, z, rotX, rotY, rotZ, ns] stiffness
    self.activeControl = True
    time.sleep(1)

    rospy.Subscriber("/cartesian_pose", PoseStamped, self.ee_pos_callback)
    rospy.Subscriber("/joint_states", JointState, self.read_conf)
    rospy.Subscriber("/force_torque_ext", WrenchStamped , self.read_force)
    # attractor publisher
    time.sleep(1)

    self.setForceTransform(lambda f,t:f)

  def setForceTransform(self, transform:np.ndarray):
    self.forceAtHuman = transform

  def reach_goal(self, trajectory:ReferenceTrajectory, varStiff:bool=True, debugLevel:int=1):
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
    T_reduce_stiff = 1.   # time of human resistance to drop stiffness [s]

    trajectory.initialize(self.curr_pos, self.curr_ori)

    def isAtGoal(relPrecision:float=1.):
      diffX, diffQ = trajectory.getDiff2Goal(self.curr_pos, self.curr_ori)
      return diffX < xMargin*relPrecision and diffQ < qMargin*relPrecision

    posR0 = self.curr_pos; oriR0 = self.curr_ori
    posRg, oriRg = trajectory.getRobotGoal()
    def getTrajectoryStiffness(K_null:float=K_null_max):
      if not self.nullspaceControl:
        return K_lin_max, K_ang_max, 0.

      d_lin_min = 0.01;     d_lin_max = 0.1
      d_lin = min(np.linalg.norm(self.curr_pos-posRg),np.linalg.norm(self.curr_pos-posR0))
      alpha_lin = 1 if d_lin <= d_lin_min else \
                  0 if d_lin >= d_lin_max else \
                  (d_lin_max - d_lin)/(d_lin_max - d_lin_min)

      d_ang_min = np.pi/36; d_ang_max = np.pi/9
      d_ang = min(Q.rotation_intrinsic_distance(self.curr_ori,oriRg), Q.rotation_intrinsic_distance(self.curr_ori,oriR0))
      alpha_ang = 1 if d_ang <= d_ang_min else \
                  0 if d_ang >= d_ang_max else \
                  (d_ang_max - d_ang)/(d_ang_max - d_ang_min)

      alpha = min(alpha_lin, alpha_ang)
      K_lin = K_lin_min + alpha*(K_lin_max - K_lin_min)
      K_ang = K_ang_min + alpha*(K_ang_max - K_ang_min)

      if np.linalg.norm(self.curr_pos-posRg) < d_lin_min*2 and \
         Q.rotation_intrinsic_distance(self.curr_ori,oriRg) < d_ang_min*2:
        K_null = max(K_null_min, K_null - dt/.5*(K_null_max-K_null_min))
      else:
        K_null = min(K_null_max, K_null + dt/.5*(K_null_max-K_null_min))
      return K_lin, K_ang, K_null

    # control loop
    dt = 1/30   # loop_time = 1/control_frequency
    r = rospy.Rate(1/dt)
    
    goal = PoseStamped()
    jointConfiguration = Float32MultiArray()
    stiff_des = Float32MultiArray()

    stiff_null = K_null_min
    interactionStiffness=1.
    noMotionIter = 0
    old_pose = [self.curr_pos, self.curr_ori]

    self.activeControl = True
    if debugLevel >= 1:
      print("Active control switched on")

    while not isAtGoal():
      # compute setpoint
      xSetpoint, qSetpoint = trajectory.getSetpoint(dt, self.curr_pos, self.curr_ori)

      if debugLevel >= 1:
        if (err:=np.linalg.norm(self.force)) > fErr:
          print('High force detected: {0} N'.format(err))
        if (err:=np.linalg.norm(self.torque)) > tErr:
          print('High torque detected: {0} Nm'.format(err))
        if (err:=np.linalg.norm(xSetpoint-self.curr_pos)) > xErr:
          print('Large distance error: {0} m'.format(err))
        if (err:=Q.rotation_intrinsic_distance(qSetpoint, self.curr_ori)) > angErr:
          print('Large angular error: {0} rad'.format(err))

      stiff_lin, stiff_ang, stiff_null = getTrajectoryStiffness(stiff_null)
      if varStiff:
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
      goal.pose.position.x = xSetpoint[0]
      goal.pose.position.y = xSetpoint[1]
      goal.pose.position.z = xSetpoint[2]

      goal.pose.orientation.x = qSetpoint.x
      goal.pose.orientation.y = qSetpoint.y
      goal.pose.orientation.z = qSetpoint.z
      goal.pose.orientation.w = qSetpoint.w

      if debugLevel >= 2:
        print("Goal:")
        print(goal.pose)
      self.goal_pub.publish(goal)

      if self.nullspaceControl and self.activeControl:
        goalPoseArray = np.hstack([xSetpoint, Q.as_float_array(qSetpoint)])
        configuration = self.gp_null.predict([goalPoseArray]) + self.gp_null0
        configuration = np.minimum(np.maximum(configuration, self.minJoints), self.maxJoints)

        if np.max(np.abs(configuration - self.curr_conf)) > 0.5:
          stiff_null = min(stiff_null, 2.)

        jointConfiguration.data = configuration[0].astype(np.float32)
        if debugLevel >= 2:
          print("Joint configuration:")
          print(jointConfiguration.data)
        self.q_pub.publish(jointConfiguration)

      if self.eeDOF < 4:
        stiff_ang = 25.
      self.stiffness = np.array([stiff_lin, stiff_lin, stiff_lin, 25., 25., stiff_ang, stiff_null])
      stiff_des.data = self.stiffness.astype(np.float32)
      if debugLevel >= 2:
        print("Stiffness:")
        print(stiff_des)
      self.stiff_pub.publish(stiff_des)

      #use rate of ROS
      r.sleep()

      # exit when stiffness is dropped
      if stiff_lin == 0 and (self.eeDOF < 4 or stiff_ang == 0):
        self.activeControl = False
        if debugLevel >= 1:
          print("Active control switched off because of zero stiffness")
        # self.stiffness = np.array([0., 0., 0., 25., 25., 0., 0.])
        return 1

      # exit when standing still
      if np.linalg.norm(self.curr_pos - old_pose[0]) < xMargin and Q.rotation_intrinsic_distance(self.curr_ori, old_pose[1]) < qMargin:
        noMotionIter += 1
        if noMotionIter > 0.5/dt: # 0.5 sec
          return 0 if isAtGoal(10) else 2
      else:
        noMotionIter = 0
      old_pose = [self.curr_pos, self.curr_ori]

    return 0

  def setPassive(self, debugLevel=0):
    self.activeControl = False
    if debugLevel >= 1:
      print("Active control switched off in passive mode")

    if not self.nullspaceControl:
      self.stiffness = np.array([0.0, 0.0, 0.0, 25.0, 25.0, 25.0 if self.eeDOF<4 else 0.0, 0.0])
      stiff_des = Float32MultiArray()
      stiff_des.data = self.stiffness.astype(np.float32)
      # if debugLevel >= 1:
      #   print("Stiffness:")
      #   print(stiff_des)
      self.stiff_pub.publish(stiff_des)
    return 0

  def get_current_pose(self) -> np.array:
    return self.curr_pos, Q.as_float_array(self.curr_ori)

#%%
