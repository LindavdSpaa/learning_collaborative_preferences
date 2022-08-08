"""
By Linda van der Spaa, 2022
l.f.vanderspaa@tudelft.nl
"""

import numpy as np
import quaternion as Q

from ReferenceTrajectories import ReferenceTrajectory
from WardrobeScenarioModel import WardrobeScenario

scenario = WardrobeScenario(None)

tf = scenario.getRobotPoseWrtObject
tfInv = scenario.getObjectPoseWrtRobot
reachState = lambda pos,ori: ReferenceTrajectory(pos,ori, tf,tfInv)
reachDirect = lambda pos,ori: ReferenceTrajectory(pos,ori)

positions = scenario.statesSet.data[:,:3]
orientations = scenario.statesSet.data[:,3:7]
qori0 = Q.from_float_array(orientations[0])

abstrActions = scenario.actionsSet

stateActionTrajectorySet = {}
for sIdx, actionList in enumerate(abstrActions):
  trajectories = []  # no trajectory for passive
  aIdx = 1
  if sIdx < 12:
    # no trajectory for grasp and let go
    aIdx += 1
    # take off support
    if sIdx == 3:
      stateActionTrajectorySet[(sIdx, aIdx)] = reachState(np.vstack([np.append(positions[3][:2],
                                                                  positions[12][2:3]),
                                                        positions[12]]),
                                              np.vstack([orientations[3],
                                                          orientations[12]]))
      aIdx += 1
    elif sIdx == 7:
      stateActionTrajectorySet[(sIdx, aIdx)] = reachState(np.vstack([np.append(positions[7][:2],
                                                                    positions[19][2:3]),
                                                          positions[19]]),
                                              np.vstack([orientations[7],
                                                          orientations[19]]))
      aIdx += 1
    elif sIdx == 11:
      stateActionTrajectorySet[(sIdx, aIdx)] = reachState(np.vstack([np.append(positions[11][:2],
                                                                    positions[23][2:3]),
                                                          positions[23]]),
                                              np.vstack([orientations[11],
                                                          orientations[23]]))
      aIdx += 1
  else:
    # put on support
    if sIdx == 12:
      stateActionTrajectorySet[(sIdx, aIdx)] = reachState(np.vstack([positions[12],
                                                          np.append(positions[3][:2],   
                                                                    positions[12][2:3]),
                                                          positions[3]]),
                                              np.vstack([orientations[12],
                                                          orientations[3],
                                                          orientations[3]]))
      aIdx += 1
    elif sIdx == 19:
      stateActionTrajectorySet[(sIdx, aIdx)] = reachState(np.vstack([positions[19],
                                                          np.append(positions[7][:2],   
                                                                    positions[19][2:3]),
                                                          positions[7]]),
                                              np.vstack([orientations[19],
                                                          orientations[7],
                                                          orientations[7]]))
      aIdx += 1
    elif sIdx == 23:
      stateActionTrajectorySet[(sIdx, aIdx)] = reachState(np.vstack([positions[23],
                                                          np.append(positions[11][:2],   
                                                                    positions[23][2:3]),
                                                          positions[11]]),
                                              np.vstack([orientations[23],
                                                          orientations[11],
                                                          orientations[11]]))
      aIdx += 1
    # move over
    if 12 <= sIdx < 20:
      stateActionTrajectorySet[(sIdx, aIdx)] = reachState([positions[20+sIdx%4]], [orientations[sIdx]])
      aIdx += 1
    if 12 <= sIdx < 16 or 20 <= sIdx < 24:
      stateActionTrajectorySet[(sIdx, aIdx)] = reachState([positions[16+sIdx%4]], [orientations[sIdx]])
      aIdx += 1
    if 16 <= sIdx < 24:
      if sIdx%2 == 1:
        stateActionTrajectorySet[(sIdx, aIdx)] = reachDirect([tf(positions[12+sIdx%4], qori0)[0]], [orientations[sIdx]])
        aIdx += 1
      else:
        stateActionTrajectorySet[(sIdx, aIdx)] = reachState([positions[12+sIdx%4]], [orientations[sIdx]])
        aIdx += 1
    # move up/down
    if sIdx%4 < 2:
      if sIdx < 16:
        stateActionTrajectorySet[(sIdx, aIdx)] = reachDirect([tf(positions[sIdx+2], qori0)[0]], [orientations[sIdx]])
        aIdx += 1
      else:
        stateActionTrajectorySet[(sIdx, aIdx)] = reachState([positions[sIdx+2]], [orientations[sIdx]])
        aIdx += 1
    else:
      if sIdx < 16:
        stateActionTrajectorySet[(sIdx, aIdx)] = reachDirect([tf(positions[sIdx-2], qori0)[0]], [orientations[sIdx]])
        aIdx += 1
      else:
        stateActionTrajectorySet[(sIdx, aIdx)] = reachState([positions[sIdx-2]], [orientations[sIdx]])
        aIdx += 1
    # rotate
    if sIdx < 16:
      stateActionTrajectorySet[(sIdx, aIdx)] = reachDirect([tf(positions[sIdx], qori0)[0]], \
                                                [orientations[sIdx-1 if sIdx%2 else sIdx+1]])
      aIdx += 1
    else:
      stateActionTrajectorySet[(sIdx, aIdx)] = reachState([positions[sIdx-1 if sIdx%2 else sIdx+1]], \
                                               [orientations[sIdx-1 if sIdx%2 else sIdx+1]])
      aIdx += 1

goalTrajectorySet = {}
for gIdx in range(3):
  goalState = scenario.statesSet.data[4*gIdx]
  goalTrajectorySet[gIdx] = ReferenceTrajectory([goalState[:3]],[goalState[3:7]], tf,tfInv)