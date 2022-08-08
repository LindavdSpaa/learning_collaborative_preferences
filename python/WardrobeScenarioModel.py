"""
By Linda van der Spaa, 2022
l.f.vanderspaa@tudelft.nl
"""

import numpy as np
import quaternion as Q
from ScenarioModel import TwoActorWorldModel

def angleDifference(a0:np.array, a1:np.array):
  q0 = Q.from_float_array(a0)
  q1 = Q.from_float_array(a1)
  return Q.rotation_intrinsic_distance(q0, q1)

class WardrobeScenario(TwoActorWorldModel):
  def __init__(self, rcf) -> None:
    self.RCf = rcf
    offsetH = 0.05

    lHanger = 0.4   # wood: 0.4, plastic: 0.38
    worldRotationMatrix = np.array([[0,0,1],[-1,0,0],[0,-1,0]])
    self.hangerOffsetR = worldRotationMatrix.dot(np.array([0,lHanger/2,0]))

    h_bar = 0.82    # wood: 0.81; plastic: 0.775
    y_bar1 = 0.61
    x_bar1 = 0.46
    y_bar2 = 0.625
    x_bar2 = 0.2
    ori_bar = np.array([.5, -.5, -.5, -.5])
    ori_bar /= np.linalg.norm(ori_bar)
    q_bar = Q.from_float_array(ori_bar)
    offsetXY_bar = 0.22
    offset_bar = np.matmul(Q.as_rotation_matrix(q_bar), worldRotationMatrix.dot(np.array([0, offsetXY_bar, offsetH])))

    h_peg = 0.535
    x_peg = 0.71
    y0 = -0.525
    ori_peg = np.array([0.05, -0.7, -0.05, -0.7])
    ori_peg /= np.linalg.norm(ori_peg)
    q_peg = Q.from_float_array(ori_peg)
    offsetXY_peg = 0.22
    offset_peg = np.matmul(Q.as_rotation_matrix(q_peg), worldRotationMatrix.dot(np.array([offsetXY_peg, 0, offsetH])))

    supportPoints = np.array([[x_peg, y0, h_peg, *ori_peg],
                              [x_bar1, y_bar1, h_bar, *ori_bar],
                              [x_bar2, y_bar2, h_bar, *ori_bar]])
    offsets = np.array([np.append(offset_peg, np.zeros(4)),
                        np.append(offset_bar, np.zeros(4)),
                        np.append(offset_bar, np.zeros(4))])

    self.comfHeight = h_peg + offset_peg[2]
    super().__init__(supportPoints, offsets, self.comfHeight)

  def getStateFeatures(self, state, intention, angDiff=angleDifference):
    return super().getStateFeatures(state, intention, angDiff)

  def getRobotPoseWrtObject(self, pos:np.ndarray, q:Q.numpy_quaternion):
    xyzR = pos + Q.as_rotation_matrix(q).dot(self.hangerOffsetR)
    return xyzR, q

  def getObjectPoseWrtRobot(self, posR:np.ndarray, qR:Q.numpy_quaternion):
    xyz = posR + Q.as_rotation_matrix(qR).dot(-self.hangerOffsetR)
    return xyz, qR

  def getRobotPoseInState(self, state:np.ndarray):
    # Project the states otherwise out of reach
    q = Q.from_float_array(state[3:7] if state[1] > -0.2 else self.statesSet.data[0,3:7])
    posR,_ = self.getRobotPoseWrtObject(state[:3], q)
    return posR, state[3:7]

  def getStateFromRobotPose(self, posR:np.ndarray, oriR:np.ndarray, grasp:int):
    # Project the states otherwise out of reach
    qR = Q.from_float_array(oriR if posR[1] > -0.2 else self.statesSet.data[0,3:7])
    xyz,_ = self.getObjectPoseWrtRobot(posR, qR)
    return np.concatenate((xyz, oriR, [grasp]), axis=0)

  def findClosestState(self, posR:np.array, oriR:np.array, grasp:int) -> np.array:
    state = self.getStateFromRobotPose(posR, oriR, grasp)
    d, closestPosIdx = self.statesPosSet.query(np.concatenate([state[:3],np.zeros(4)]))
    if closestPosIdx < 12:
      closestPosIdx -= closestPosIdx%4
    sIdx = closestPosIdx
    dAng0 = angleDifference(state[3:7], self.statesSet.data[sIdx,3:7])
    if sIdx < 12 and dAng0 > np.linalg.norm(state[:3]-self.statesSet.data[sIdx+(12 if sIdx<4 else 14),:3]):
      sIdx += 12 if sIdx<4 else 14
    if sIdx >= 12:
      if sIdx%2 == 1:
        sIdx -= 1
        dAng0 = angleDifference(state[3:7], self.statesSet.data[sIdx,3:7])
      dAng1 = angleDifference(state[3:7], self.statesSet.data[sIdx+1,3:7])
      if dAng1 < dAng0:
        sIdx += 1
      elif closestPosIdx < 12 and sIdx != closestPosIdx:
        sIdx = closestPosIdx
    if sIdx < 12:
      sIdx += grasp
    closestState = self.statesSet.data[sIdx]
    dLin = np.linalg.norm(closestState[:3]-state[:3])
    dAng = angleDifference(closestState[3:7],state[3:7])
    return closestState, sIdx, dLin, dAng

  def findClosestSupportIdx(self, posR:np.array, oriR:np.array):
    state = self.getStateFromRobotPose(posR, oriR, 0)
    distance, closestSupportIdx = self.supportSet.query(state[:-1])
    return closestSupportIdx, distance

  def torqueR2forceH(self, torqueR:np.ndarray):
    return np.cross(torqueR,[0,0,-1])/np.linalg.norm(2*self.hangerOffsetR)
