"""
By Linda van der Spaa, 2022
l.f.vanderspaa@tudelft.nl
"""

import copy
import numpy as np
from scipy.spatial import KDTree

class TwoActorWorldModel:
  # All directions are named from the human point of view!
  def __init__(self, supportPoints:np.array, offsets:np.array, comfHeight:float) -> None:
    offsetPoints = supportPoints + offsets
    xyPositions = offsetPoints[:,:2]
    zPositions = np.unique(np.append(offsetPoints[:,2], comfHeight))
    self.orientations = np.unique(offsetPoints[:,3:], axis=0)

    self.graspSet = list(range(4))  # ['no', 'H', 'R', 'both']
    self.supportSet = KDTree(supportPoints)
    self.offsetSet = KDTree(offsetPoints)
    self.statesSet = KDTree(np.stack([np.concatenate((p,[i])) for p in supportPoints for i in self.graspSet] +\
                                     [np.concatenate((xy,[z],o,[self.graspSet[-1]])) for xy in xyPositions for z in zPositions for o in self.orientations]))
    self.statesPosSet = KDTree(np.stack([np.concatenate((p[:3],np.zeros(4))) for p in supportPoints for i in self.graspSet] +\
                                        [np.concatenate((xy,[z],np.zeros(4))) for xy in xyPositions for z in zPositions for _ in self.orientations]))

    # Actions:
    # wait (remain passive), grasp/let go, dock/undock, move to (x,y) or h, rotate
    self.actionsSet = [self.validActionsInState(s) for s in self.statesSet.data]
    self.nActions = max([len(sa) for sa in self.actionsSet])

    # Intentions
    self.intentionsSet = self.supportSet
    self.intentionNames = [str(i) for i in range(self.intentionsSet.n)]


  def validActionsInState(self, state):
    actions = ['wait']
    if self.supportSet.query(state[:-1])[0] < 1e-6:
      actions.append('(un)grasp')
    if int(round(state[-1])) == 3:
      if self.supportSet.query(state[:-1])[0] < 1e-6:
        actions.append('take off support')
      else:
        if self.offsetSet.query(state[:-1])[0] < 1e-6:
          actions.append('put on support')
        freePoints = self.statesSet.data[self.supportSet.n*len(self.graspSet):]
        freePointsAtSameHeight = np.unique(freePoints[np.abs(freePoints[:,2]-state[2]) < 1e-9,:2], axis=0)
        for p in freePointsAtSameHeight:
          if np.linalg.norm(state[:2]-p) > 1e-3:
            actions.append('move over to {}'.format(KDTree(self.supportSet.data[:,:2]).query(p)[1]))
        for i, so in enumerate(self.offsetSet.data):
          if not np.isclose(self.offsetSet.data[:i,2:3], so[2]).any():
            if state[2] < so[2]-1e-9:
              actions.append('move up to {}'.format(i))
            elif state[2] > so[2]+1e-9:
              actions.append('move down to {}'.format(i))
        if state[2] > self.comfHeight+1e-9 and not np.isclose(self.offsetSet.data[:,2:3], self.comfHeight).any():
          actions.append('move all down')
        if len(state) > 4:
          actions.append('rotate')
    return actions

  def initPolicyActionValues(self):
    actionValues = [[np.zeros(len(self.actionsSet[sIdx]))
                     for sIdx in range(self.statesSet.n)]
                    for _ in range(self.intentionsSet.n)]
    return actionValues

  def stateTransition(self, s, ar, ah):
    def xyFromIdx(h, gIdx):
      freePoints = self.statesSet.data[self.supportSet.n*len(self.graspSet):,:-1]
      freePointsAtSameHeight = np.unique(freePoints[np.abs(freePoints[:,2]-h) < 1e-9,:2], axis=0)
      return freePointsAtSameHeight[KDTree(freePointsAtSameHeight).query(self.supportSet.data[gIdx,:2])[1]]

    sNew = copy.deepcopy(s)

    validActions = self.validActionsInState(sNew)
    if ar == '(un)grasp' and ar in validActions:
      sNew[-1] = (int(round(sNew[-1]))+2) % 4
      validActions = self.validActionsInState(sNew)
    if ah == '(un)grasp' and ah in validActions:
      graspIdx = int(round(sNew[-1]))
      sNew[-1] = (graspIdx +(3 if graspIdx%2 else 1)) % 4
      validActions = self.validActionsInState(sNew)

    if ar not in validActions or ah not in validActions:
      return (sNew, 'aborted')

    # cooperative object manipulation
    if int(round(sNew[-1])) == 3:
      action = ar.split() if ah == 'wait' else ah.split()
      if action[0] == 'move':
        if action[1] == 'over':
          sNew[:2] = xyFromIdx(sNew[2], int(action[3]))
        else:
          if action[1] == 'up':
            sNew[2] = self.offsetSet.data[int(action[3]),2]
          elif action[1] == 'down':
            sNew[2] = self.offsetSet.data[int(action[3]),2]
          elif action[2] == 'down':
            sNew[2] = np.min(self.statesSet.data[self.supportSet.n*len(self.graspSet):,2])
      elif action[0] == 'take':
        sNew[:-1] = self.offsetSet.data[self.supportSet.query(sNew[:-1])[1]]
      elif action[0] == 'put':
        sNew[:-1] = self.supportSet.data[self.offsetSet.query(sNew[:-1])[1]]
      elif action[0] == 'rotate' and (nOri := len(self.orientations)):
        oriIdx = np.linalg.norm(self.orientations-sNew[3:-1], axis=1).argmin()
        sNew[3:-1] = self.orientations[(oriIdx+1)%nOri]

    return (sNew, 'ok')

  def isSupportState(self, s):
    # object resting, robot and human free, regardless of intention
    return True if ( int(round(s[-1])) == 0 and self.supportSet.query(s[:-1])[0] < 1e-6 ) \
        else False

  def cooperationReward(self, s1, s0, ar, ah, tFlag):
    reward = 0
    cWrong = 1.0
    cWait = cWrong/self.RCf

    validActions = self.validActionsInState(s0)
    if tFlag == 'aborted' or ar not in validActions or ah not in validActions:
      reward -= cWrong
      return reward

    if ah == 'wait' and np.linalg.norm(s1[:-1] - s0[:-1]) < 1e-6:
      reward -= cWrong

    if int(round(s1[-1])) == 3:
      if ah != 'wait' and ar != ah:
        if ar == 'wait':
          reward -= cWait
        elif len(ar.split()) == len(ah.split()) == 4 and \
            ar[1] == ah[1] in ['up', 'down'] and \
            abs(self.offsetSet.data[int(ar[-1]),2] - self.offsetSet.data[int(ah[-1]),2]) < 1e-3:
          pass
        else:
          reward -= cWrong

    return reward

  def setStartState(self, sIdx=0):
    self.startState = np.append(self.supportSet.data[sIdx],0)

  def getStartStateBelief(self):
    sIdx = self.supportSet.query(self.startState[:-1])[1]
    return self.startState, [1/(self.intentionsSet.n-1) if i != sIdx else 0 for i in range(self.intentionsSet.n)]

  ## features for human reward model
  # defined relative to:  - intended goal state, 
  #                       - other possiblie final states, 
  #                       - some absolute world values: comfHeight, orientation
  def getStateFeatures(self, state, intention, angDiff):
    def getHorDistance(searchTree4D, xy):
      return KDTree(searchTree4D.data[:,:2]).query(xy)[0]
    def getVertDistance(searchTree4D, z):
      return KDTree(searchTree4D.data[:,2:3]).query(z)[0]

    def rbfValX(d):     # linear radial basis function
      w = 0.02
      return np.exp(-d*d/(2*w*w))
    def rbfValAng(d):     # angular radial basis function
      w = np.pi/18
      return np.exp(-d*d/(2*w*w))

    phiSet = np.append(self.orientations, [intention[3:]], axis=0)
    graspState = int(round(state[-1]))

    # At the intended goal position
    diffXYZPhiIntention = state[:-1]-self.offsetSet.data[self.supportSet.query(intention)[1]]
    xyToIntentionOffset = np.linalg.norm(diffXYZPhiIntention[:2])
    zToIntentionOffset = abs(diffXYZPhiIntention[2])
    featureArray = np.array([rbfValX(np.linalg.norm(state[:3]-intention[:3]))*rbfValAng(angDiff(state[3:-1],intention[3:])) if graspState == grasp else 0. for grasp in self.graspSet])
    featureArray = np.append(featureArray, [rbfValX(xyToIntentionOffset)*rbfValX(zToIntentionOffset)*rbfValAng(angDiff(state[3:-1],phi)) for phi in phiSet])
    featureArray = np.append(featureArray, [rbfValX(xyToIntentionOffset)*rbfValX(state[2]-self.comfHeight)*rbfValAng(angDiff(state[3:-1],phi)) for phi in phiSet])

    # Alternative goals
    nonGoalsSet = KDTree(np.delete(self.supportSet.data, self.supportSet.query(intention)[1], axis=0))
    closestNonGoal = nonGoalsSet.data[nonGoalsSet.query(state[:-1])[1]]
    nonGoalsOffsetSet = KDTree(np.delete(self.offsetSet.data, self.supportSet.query(intention)[1], axis=0))
    xyToNonGoalOffset = getHorDistance(nonGoalsOffsetSet,state[:2])
    zToNonGoalOffset = getVertDistance(nonGoalsOffsetSet,state[2])

    featureArray = np.append(featureArray, [rbfValX(np.linalg.norm(state[:3]-closestNonGoal[:3]))*rbfValAng(angDiff(state[3:-1],closestNonGoal[3:])) if graspState == grasp else 0. for grasp in self.graspSet])
    featureArray = np.append(featureArray, [rbfValX(xyToNonGoalOffset)*rbfValX(zToNonGoalOffset)*rbfValAng(angDiff(state[3:-1],phi)) for phi in phiSet])
    featureArray = np.append(featureArray, [rbfValX(xyToNonGoalOffset)*rbfValX(state[2]-self.comfHeight)*rbfValAng(angDiff(state[3:-1],phi)) for phi in phiSet])

    # Intermediate states
    featureArray = np.append(featureArray, [rbfValX(xyToIntentionOffset)*rbfValX(zToNonGoalOffset)*rbfValAng(angDiff(state[3:-1],phi)) for phi in phiSet])
    featureArray = np.append(featureArray, [rbfValX(xyToNonGoalOffset)*rbfValX(zToIntentionOffset)*rbfValAng(angDiff(state[3:-1],phi)) for phi in phiSet])

    if (featureSum := np.sum(featureArray)) > 1:
      return featureArray/featureSum
    elif featureSum < 1e-9:
      print("Warning: No features in this state!")
    return featureArray

  def getFeatureMatrix(self):
    featureSet = [self.getStateFeatures(s, i) for i in self.intentionsSet.data for s in self.statesSet.data]
    return np.stack(featureSet)

  def getIntention(self, idx):
    return self.intentionsSet.data[idx]

  def getActionSet(self, stateIdx):
    return self.actionsSet[stateIdx]

  def getAction(self, stateIdx, actionIdx):
    return self.actionsSet[stateIdx][actionIdx]

  def getStateIdx(self, state):
    _, stateIdx = self.statesSet.query(state)
    return stateIdx

  def getActionIdx(self, stateIdx, action):
    return self.actionsSet[stateIdx].index(action)

  def getIntentionIdx(self, intention):
    _, intentionIdx = self.intentionsSet.query(intention)
    return intentionIdx


  ## Single-Agent Transition and Reward matrices
  # in: partner                     --  "human", "robot"
  #     partnerActionProbabilities  --  [[np.array()]] shape (nIntentions, nStates)
  #                                     variable array length: nLocalActions
  #     partnerReward               --  np.array() shape (nIntentions, nStates)
  # out: transitions T([s,i],a,[s',i']) --  np.array() shape (nIntentions x nStates, nActions (=max nLocalActions), nIntentions x nStates))
  #      rewards R([s,i],a,[s',i'])     --  np.array() shape (nIntentions x nStates, nActions (=max nLocalActions), nIntentions x nStates)
  def getMatricesTR(self, actor, partnerActionProbabilities, partnerReward=None):
    nI = self.intentionsSet.n
    nS = self.statesSet.n
    nA = self.nActions

    computeReward = partnerReward is not None

    transitionFunc = lambda s,a,ap: self.stateTransition(s,a,ap) if actor=='robot' else self.stateTransition(s,ap,a)
    if computeReward:
      coopReward = lambda s1,s0,a,ap,f: self.cooperationReward(s1,s0,a,ap,f) if actor=='robot' else self.cooperationReward(s1,s0,ap,a,f)

    # Transitions T(s,a,s') and rewards R(s,a,s')
    T = np.zeros((nI*nS, nA, nI*nS))
    R = np.zeros((nI*nS, nA, nI*nS))

    for iIdx in range(nI):
      intention = self.intentionsSet.data[iIdx]
      for s0Idx in range(nS):
        s0iIdx = nS*iIdx + s0Idx
        s0 = self.statesSet.data[s0Idx]

        for aIdx in range(nA):
          # Terminal state
          if np.linalg.norm(s0[:-1]-intention) < 1e-3 and int(round(s0[-1])) == 0:
            T[s0iIdx, aIdx, s0iIdx] = 1.0
          else:
            localActionSet = self.actionsSet[s0Idx]
            nLocalActions = len(localActionSet)
            a = localActionSet[aIdx % nLocalActions]

            for apIdx, ap in enumerate(localActionSet):
              pPartnerAction = partnerActionProbabilities[iIdx][s0Idx][apIdx]
              (s1,flag) = transitionFunc(s0,a,ap)

              s1iIdx = nS*iIdx + self.statesSet.query(s1)[1]
              T[s0iIdx, aIdx, s1iIdx] += pPartnerAction

              if computeReward:
                reward = 0.#self.goalReward(s0, s1)
                reward += coopReward(s1, s0, a, ap, flag)
                reward += partnerReward[iIdx, s0Idx]
                R[s0iIdx, aIdx, s1iIdx] += pPartnerAction*reward

    return (T, R) if computeReward else T

  ## Prune trajectory trace to shortest path in observed trajectory
  # Inputs: - stateActionTrace -- list of [s, b(i), ar, ah, s']
  #         - intentionIdx -- int in range(len(self.intentionsSet))
  # Output: - pruned trajectory -- list of [siIdx, ah, si'Idx]
  def pruneTrajectoryTrace(self, stateActionsTrace, intentionIdx):
    traceLength = len(stateActionsTrace)
    trajectory = np.zeros((1,traceLength+1,3), int)
    for i in range(traceLength):
      state = stateActionsTrace[i][0]
      _, stateIdx = self.statesSet.query(state)

      ah = stateActionsTrace[i][2]
      ahIdx = self.actionsSet[stateIdx].index(ah)

      nextState = stateActionsTrace[i][4]
      nextStateIdx = self.statesSet.query(nextState)[1]

      si0Idx = self.statesSet.n*intentionIdx + stateIdx
      si1Idx = self.statesSet.n*intentionIdx + nextStateIdx
      trajectory[0,i] = [si0Idx, ahIdx, si1Idx]
    # Adding the final state
    trajectory[0,-1,0] = self.statesSet.n*intentionIdx + self.statesSet.query(stateActionsTrace[-1][4])[1]

    # Remove loops that return to earlier states, necessary in order to stabilize learning process
    i = 1
    while i < trajectory.shape[1]:
      j = 1
      while j <= i:
        if trajectory[0, i, 0] == trajectory[0, i-j, 0]:
          trajectory = np.delete(trajectory, list(range(i-j,i)), axis=1)
          i-=j
        j+=1
      i+=1

    return trajectory

  def getHumanActionFromStateTransition(self, s0, ar, s1):
    action = ar
    sNew,_ = self.stateTransition(s0, ar, action)
    if np.allclose(s1, sNew):
      pass
    elif np.allclose(s1[:-1], s0[:-1]):
      gh0 = int(round(s0[-1])) % 2
      gh1 = int(round(s1[-1])) % 2
      action ='(un)grasp' if gh1 != gh0 else 'wait'
    elif self.supportSet.query(s0[:-1])[0] < 1e-9:
      action = 'take off support'
    else:
      d0, supportIdx0 = self.offsetSet.query(s0[:-1])
      d1, supportIdx1 = self.supportSet.query(s1[:-1])
      if d0 < 1e-9 and d1 < 1e-9 and supportIdx1 == supportIdx0:
        action = 'put on support'
      else:
        s1a = self.offsetSet.data[supportIdx1] if d1 < 1e-9 else s1
        diffxy = np.linalg.norm(s1a[:2] - s0[:2])
        diffz = np.abs(s1a[2] - s0[2])
        if diffxy < 1e-9 and diffz < 1e-9:
          action = 'rotate'
        elif diffxy > 1e-9 and not (1e-9 < diffz < diffxy):
          action = 'move over to {}'.format(KDTree(self.offsetSet.data[:,:2]).query(s1a[:2])[1])
        else:
          if np.abs(s1a[2] - self.comfHeight) < 1e-9 and not np.isclose(self.offsetSet.data[:,2:3], self.comfHeight).any():
            action = 'move all down'
          else:
            hIdx = KDTree(self.offsetSet.data[:,2:3]).query(s1a[2:3])[1]
            action = 'move up to {}'.format(hIdx) if s1a[2] > s0[2] else 'move down to {}'.format(hIdx)
    sNew,_ = self.stateTransition(s0, ar, action)
    return action, sNew

  def inferHumanAction(self, s0, s1, ar):
    stateSequence = [s0]
    robotActionSequence = [ar]
    humanActionSequence = []
    for i, robotAction in enumerate(robotActionSequence):
      state = stateSequence[i]
      humanAction, nextState = self.getHumanActionFromStateTransition(state, robotAction, s1)
      humanActionSequence.append(humanAction)
      if not np.allclose(s1, nextState):
        stateSequence.insert(i+1, nextState)
        robotActionSequence.insert(i+1, 'wait')
      if i == 10:   # to avoid infinite looping
        print("Warning: no valid human action sequence found!")
        break
    stateSequence.append(s1)
    return stateSequence, robotActionSequence, humanActionSequence

  # Baseline policies for comparing performance
  def getPassiveAction(self, state):
    if self.supportSet.query(state[:-1])[0] < 1e-9 and \
        int(round(state[-1])) in ([2, 3] if np.linalg.norm(self.startState[:3]-state[:3]) > 1e-3 else [0, 1]):
      return '(un)grasp'
    else:
      return 'wait'

  def getImitatedAction(self, state):
    if not hasattr(self, 'stateActionImitationList'):
      return self.getPassiveAction(state)
    elif (stateKey := tuple([self.supportSet.query(self.startState[:-1])[1], self.statesSet.query(state)[1]])) \
        not in self.stateActionImitationList:
      return self.getPassiveAction(state)
    else:
      candidateActions = self.stateActionImitationList[stateKey]
      return max(set(candidateActions), key=candidateActions.count)

  def updateActionImitationList(self, state, ah, bufferSize=1):
    if not hasattr(self, 'stateActionImitationList'):
      self.stateActionImitationList = {}

    if ah != 'wait':
      stateKey = tuple([self.supportSet.query(self.startState[:-1])[1], self.statesSet.query(state)[1]])
      if stateKey not in self.stateActionImitationList:
        self.stateActionImitationList[stateKey] = []
      elif len(self.stateActionImitationList[stateKey]) == bufferSize:
        self.stateActionImitationList[stateKey].pop(-1)
      self.stateActionImitationList[stateKey].append(ah)