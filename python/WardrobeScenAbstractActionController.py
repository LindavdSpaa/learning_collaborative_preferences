"""
By Linda van der Spaa, 2022
l.f.vanderspaa@tudelft.nl
"""

from TrajectoryImpedanceControl import ReferenceTrajectory, VariableImpedanceController
from WardrobeScenarioModel import WardrobeScenario
import quaternion as Q
import numpy as np
import time, copy, rospy
from geometry_msgs.msg import WrenchStamped, PoseStamped
from std_msgs.msg import Bool, Int8

#%%
#%%
class WardrobeAbstractActionController:
  def ee_pos_callback(self, data): 
    self.curr_pos = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
    curr_ori_arr = np.array([data.pose.orientation.w, data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z])
    if curr_ori_arr[0] < 0:
      curr_ori_arr *= -1
    self.curr_ori = curr_ori_arr

  def read_force(self, data):
    self.normForce = np.linalg.norm([data.wrench.force.x, data.wrench.force.y, data.wrench.force.z])

  def updateGraspState(self, msg):
    self.graspState = msg.data

  def __init__(self, varStiff=False) -> None:
    # Make sure a ROS node is running
    self.controller = VariableImpedanceController()

    rospy.Subscriber("/cartesian_pose", PoseStamped, self.ee_pos_callback)
    rospy.Subscriber("/force_torque_ext", WrenchStamped , self.read_force)
    rospy.Subscriber('/grasp_controller/state', Int8, self.updateGraspState)
    self.grasp_pub = rospy.Publisher('/grasp_controller/grasp', Bool, queue_size=0)

    self.scenario = WardrobeScenario(None)
    self.controller.setForceTransform(lambda f,t: f + self.scenario.torqueR2forceH(t))

    self.actionsSet = self.createControlActionSet(varStiff)

  def grasp(self, grasp:bool):
    tTimeout = 10  # wait for maximum 2s

    self.grasp_pub.publish(grasp)
    t0 = time.time()
    while time.time()-t0 < tTimeout:
      if self.graspState == (1 if grasp else 0):
        return 0
      time.sleep(0.05)
    return -1

  def createControlActionSet(self, varStiff=False):
    tf = self.scenario.getRobotPoseWrtObject
    tfInv = self.scenario.getObjectPoseWrtRobot
    reachState = lambda pos,ori: self.controller.reach_goal(ReferenceTrajectory(pos,ori, tf,tfInv), varStiff,0)
    reachDirect = lambda pos,ori: self.controller.reach_goal(ReferenceTrajectory(pos,ori), varStiff=varStiff, debugLevel=0)

    positions = self.scenario.statesSet.data[:,:3]
    orientations = self.scenario.statesSet.data[:,3:7]
    qori0 = Q.from_float_array(orientations[0])

    abstrActions = self.scenario.actionsSet

    actionsSet = []
    for sIdx,_ in enumerate(self.scenario.actionsSet):
      actions = [([abstrActions[sIdx][0], sIdx, 0], lambda s: self.controller.setPassive())]
      aIdx = 1
      if sIdx < 12:
        # grasp and let go
        actions.append(([abstrActions[sIdx][aIdx], sIdx, aIdx],
                         lambda s: self.grasp(True if s%4 < 2 else False)))
        aIdx += 1
        # take off support
        if sIdx == 3:
          actions.append(([abstrActions[sIdx][aIdx], sIdx, aIdx],
                           lambda s: reachState(np.vstack([np.append(positions[3][:2],
                                                                     positions[12][2:3]),
                                                           positions[12]]),
                                                np.vstack([orientations[3],
                                                           orientations[12]]))))
          aIdx += 1
        elif sIdx == 7:
          actions.append(([abstrActions[sIdx][aIdx], sIdx, aIdx],
                           lambda s: reachState(np.vstack([np.append(positions[7][:2],
                                                                     positions[19][2:3]),
                                                           positions[19]]),
                                                np.vstack([orientations[7],
                                                           orientations[19]]))))
          aIdx += 1
        elif sIdx == 11:
          actions.append(([abstrActions[sIdx][aIdx], sIdx, aIdx],
                           lambda s: reachState(np.vstack([np.append(positions[11][:2],
                                                                     positions[23][2:3]),
                                                           positions[23]]),
                                                np.vstack([orientations[11],
                                                           orientations[23]]))))
          aIdx += 1
      else:
        # put on support
        if sIdx == 12:
          actions.append(([abstrActions[sIdx][aIdx], sIdx, aIdx],
                           lambda s: reachState(np.vstack([positions[12],
                                                           np.append(positions[3][:2],   
                                                                     positions[12][2:3]),
                                                           positions[3]]),
                                                np.vstack([orientations[12],
                                                           orientations[3],
                                                           orientations[3]]))))
          aIdx += 1
        elif sIdx == 19:
          actions.append(([abstrActions[sIdx][aIdx], sIdx, aIdx],
                           lambda s: reachState(np.vstack([positions[19],
                                                           np.append(positions[7][:2],   
                                                                     positions[19][2:3]),
                                                           positions[7]]),
                                                np.vstack([orientations[19],
                                                           orientations[7],
                                                           orientations[7]]))))
          aIdx += 1
        elif sIdx == 23:
          actions.append(([abstrActions[sIdx][aIdx], sIdx, aIdx],
                           lambda s: reachState(np.vstack([positions[23],
                                                           np.append(positions[11][:2],   
                                                                     positions[23][2:3]),
                                                           positions[11]]),
                                                np.vstack([orientations[23],
                                                           orientations[11],
                                                           orientations[11]]))))
          aIdx += 1
        # move over
        if 12 <= sIdx < 20:
          actions.append(([abstrActions[sIdx][aIdx], sIdx, aIdx],
                           lambda s: reachState([positions[20+s%4]], [orientations[s]])))
          aIdx += 1
        if 12 <= sIdx < 16 or 20 <= sIdx < 24:
          actions.append(([abstrActions[sIdx][aIdx], sIdx, aIdx],
                           lambda s: reachState([positions[16+s%4]], [orientations[s]])))
          aIdx += 1
        if 16 <= sIdx < 24:
          if sIdx%2 == 1:
            actions.append(([abstrActions[sIdx][aIdx], sIdx, aIdx],
                             lambda s: reachDirect([tf(positions[12+s%4], qori0)[0]], [orientations[s]])))
            aIdx += 1
          else:
            actions.append(([abstrActions[sIdx][aIdx], sIdx, aIdx],
                             lambda s: reachState([positions[12+s%4]], [orientations[s]])))
            aIdx += 1
        # move up/down
        if sIdx%4 < 2:
          if sIdx < 16:
            actions.append(([abstrActions[sIdx][aIdx], sIdx, aIdx],
                             lambda s: reachDirect([tf(positions[s+2], qori0)[0]], [orientations[s]])))
            aIdx += 1
          else:
            actions.append(([abstrActions[sIdx][aIdx], sIdx, aIdx],
                             lambda s: reachState([positions[s+2]], [orientations[s]])))
            aIdx += 1
        else:
          if sIdx < 16:
            actions.append(([abstrActions[sIdx][aIdx], sIdx, aIdx],
                             lambda s: reachDirect([tf(positions[s-2], qori0)[0]], [orientations[s]])))
            aIdx += 1
          else:
            actions.append(([abstrActions[sIdx][aIdx], sIdx, aIdx],
                             lambda s: reachState([positions[s-2]], [orientations[s]])))
            aIdx += 1
        # rotate
        if sIdx < 16:
          actions.append(([abstrActions[sIdx][aIdx], sIdx, aIdx],
                           lambda s: reachDirect([tf(positions[s], qori0)[0]], [orientations[s-1 if s%2 else s+1]])))
          aIdx += 1
        else:
          actions.append(([abstrActions[sIdx][aIdx], sIdx, aIdx],
                           lambda s: reachState([positions[s-1 if s%2 else s+1]], [orientations[s-1 if s%2 else s+1]])))
          aIdx += 1
      actionsSet.append(actions)
    return actionsSet

  def waitForNextState(self, currState):
    dt = 1/30             # s time before checking again
    timeoutTime = 60      # s before stopping to check, > dt
    stateMarginLinL = .5  # m large distance considered close to a state
    stateMarginAngL = .8  # rad large distance considered close to a state
    stateMarginLinS = .03 # m small distance considered close to a state
    forceMargin = 1.5     # N normal force considered close to zero
    eps = 1e-10           # floatingpoint error considered 0

    humanGraspCorrection = 0
    waitedTime = 0
    pose = (self.curr_pos,self.curr_ori)
    t = time.time()
    for _ in range(int(timeoutTime/dt)):
      state, stateIdx, dLin, dAng = self.scenario.findClosestState(self.curr_pos,self.curr_ori, 3*self.graspState+humanGraspCorrection)

      if np.linalg.norm(state-currState) > eps and \
          ((stateIdx >= 12 and dLin <= stateMarginLinL and dAng <= stateMarginAngL) or\
          ((dLin <= stateMarginLinS or waitedTime > 0.5) and self.normForce < forceMargin)):
        return state

      time.sleep(max(0,dt + t - time.time()))
      t = time.time()
      
      waitedTime = (waitedTime + dt) if np.linalg.norm(np.hstack((self.curr_pos,self.curr_ori))-np.hstack(pose)) < 1e-3 else 0
      pose = (self.curr_pos,self.curr_ori)
      
      # after 5 seconds of inactivity, check if object is at support.
      # If so, assume human has ungrasped
      ## P.S. better to not assume human grasps in start state at t=0
      if waitedTime > 5. and np.allclose(state[-1]%2,1):
        checkState = copy.deepcopy(state)
        checkState[-1] = 0
        if self.scenario.isSupportState(checkState):
          humanGraspCorrection = -1
      # elif waitedTime < dt/2 and np.allclose(graspState%2,0):
        # graspState += 1

    return state

  def doAction(self, stateIdx, actionIdx, withGrasp=True) -> int:
    print(self.actionsSet[stateIdx][actionIdx][0])

    exitflag = self.actionsSet[stateIdx][actionIdx][1](stateIdx)
    if exitflag == 1:
      print("Dropped to passive")
      self.controller.setPassive()
      state = self.waitForNextState(self.getState(withGrasp))
      return state, self.scenario.getStateIdx(state)
    elif actionIdx == 0:
      state = self.waitForNextState(self.getState(withGrasp))
      return state, self.scenario.getStateIdx(state)
    elif exitflag == -1:
      print("Grasp action failed")
      state = self.getState(withGrasp)
      if state[-1] == self.scenario.statesSet.data[stateIdx][-1] == 3 \
        and self.scenario.isSupportState(np.append(state[:-1],0)):
        # exit the task without waiting for the letting go to succeed
        state[-1] = 0
      return state, self.scenario.getStateIdx(state)
    elif exitflag != 0:
      print("Warning: action returned unknown exit flag")
    return self.getStateAndIdx(withGrasp)

  def goToGoal(self, goalIdx):
    goalState = self.scenario.statesSet.data[4*goalIdx]
    tf = self.scenario.getRobotPoseWrtObject
    tfInv = self.scenario.getObjectPoseWrtRobot
    self.controller.reach_goal(ReferenceTrajectory([goalState[:3]],[goalState[3:7]], tf,tfInv), False,0)
    return self.getStateAndIdx()

  def goToClosestGoal(self):
    supportIdx,_ = self.scenario.findClosestSupportIdx(self.curr_pos,self.curr_ori)
    return self.goToGoal(supportIdx)

  def getState(self, checkGrasp=True):
    state = copy.deepcopy(self.scenario.findClosestState(self.curr_pos,self.curr_ori,3)[0])
    if checkGrasp:
      for _ in range(10):
        if self.graspState == 1:
          break
        elif self.graspState == 0 and self.scenario.isSupportState(np.append(state[:-1],0)):
          state[-1] = 0
          break
        else:
          print("Waiting to recover grasp...")
          self.grasp(True)
          time.sleep(1)
    return state

  def getStateIdx(self, checkGrasp=True) -> int:
    return self.scenario.getStateIdx(self.getState(checkGrasp))

  def getStateAndIdx(self, checkGrasp=True):
    state = self.getState(checkGrasp)
    return state, self.scenario.getStateIdx(state)