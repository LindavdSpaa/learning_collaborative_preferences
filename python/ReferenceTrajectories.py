"""
Piece-wise linear trajectories with modulated attractor distance following (Gams et al., 2009)

By Linda van der Spaa, 2022
l.f.vanderspaa@tudelft.nl
"""

import numpy as np
import quaternion as Q

from numpy.core.fromnumeric import mean, prod
from typing import Tuple

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
