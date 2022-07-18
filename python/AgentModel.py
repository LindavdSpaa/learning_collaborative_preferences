"""
By Linda van der Spaa, 2022
l.f.vanderspaa@tudelft.nl
"""

import copy, random
import numpy as np

from QIteration import qIteration

def softMax(vec):
  expVec = np.exp(vec)
  return expVec/np.sum(expVec)

def wrapValues(array, newLength):
  oldLength = len(array)
  newArray = np.zeros((newLength))
  if newLength == oldLength:
    newArray = np.copy(array)
  elif newLength < oldLength:
    for i in range(oldLength):
      idx = i % newLength
      newArray[idx] += array[i]
    doubleActions = np.array([int(oldLength/newLength) + (1 if i<oldLength%newLength else 0) for i in range(newLength)])
    newArray /= doubleActions
  return newArray

class AgentModel:
  def __init__(self, scenario, softMaxTemp, gamma, actionBound=0.9) -> None:
    self.temperature = softMaxTemp
    self.gamma = gamma
    self.actionBound = actionBound
    self.getBaselineAction = lambda idx: scenario.actionsSet[idx].index(scenario.getPassiveAction(scenario.statesSet.data[idx]))
    self.nMaxActions = scenario.nActions
    self.reset(scenario)

  def reset(self, scenario):
    self.QValues = scenario.initPolicyActionValues()
    self.setActionProbabilitiesPerIntention()

  def setQValuesInState(self, stateQVal, sIdx, iIdx):
    nLocalActions = len(self.QValues[iIdx][sIdx])
    self.QValues[iIdx][sIdx] = wrapValues(stateQVal, nLocalActions)
    self.QValues[iIdx][sIdx] -= max(self.QValues[iIdx][sIdx])

  def setQValues(self, QVal, intentionIdx):
    for stateIdx in range(len(self.QValues[intentionIdx])):
      self.setQValuesInState(QVal[stateIdx], stateIdx, intentionIdx)

  def setActionProbabilitiesPerIntention(self):
    if not hasattr(self, 'actionProbabilities'):
      self.actionProbabilities = copy.deepcopy(self.QValues)
    
    nIntentions = len(self.QValues)
    for iIdx in range(nIntentions):
      nStates = len(self.QValues[iIdx])
      for sIdx in range(nStates):
        self.actionProbabilities[iIdx][sIdx] = softMax(self.QValues[iIdx][sIdx]*self.temperature)

  def updatePolicy(self, transitionMat, rewardMat):
    nIntentions = len(self.QValues)
    nStates = len(self.QValues[0])

    wrappedQ = qIteration(transitionMat, rewardMat, self.gamma)
    wrappedQ = np.reshape(wrappedQ, (nIntentions,nStates,self.nMaxActions))
    for iIdx in range(nIntentions):
      self.setQValues(wrappedQ[iIdx], iIdx)
    self.setActionProbabilitiesPerIntention()

  def updatePolicyPerIntention(self, transitionMat, rewardMat):
    nIntentions = len(self.QValues)
    nStates = len(self.QValues[0])
    for iIdx in range(nIntentions):
      idx0 = iIdx*nStates
      idx1 = (iIdx+1)*nStates
      wrappedQ = qIteration(transitionMat[idx0:idx1,:,idx0:idx1],
                            rewardMat[idx0:idx1,:,idx0:idx1],
                            self.gamma)
      self.setQValues(wrappedQ, iIdx)
    self.setActionProbabilitiesPerIntention()

  def getActionDistribution(self, stateIdx, belief):
    nIntentions = len(self.QValues)
    nLocalActions = len(self.QValues[0][stateIdx]) # assumed independent of intention
    Q = np.zeros(nLocalActions)
    for iIdx in range(nIntentions):
      Q += belief[iIdx]*self.QValues[iIdx][stateIdx]
    Q -= max(Q)
    return softMax(Q*self.temperature)

  def getActionIdx(self, stateIdx, belief):
    actionDistribution = self.getActionDistribution(stateIdx, belief)

    # Only consider actions at least as good as the baseline policy and relative action probability bound
    threshold = max(actionDistribution[self.getBaselineAction(stateIdx)],
                    self.actionBound*np.max(actionDistribution))
    actionDistribution = np.array([(a if a >= threshold else 0) for a in actionDistribution])
    actionDistribution = actionDistribution/np.sum(actionDistribution)

    # Draw action from distribution
    pActions = np.cumsum(actionDistribution)
    randomNumber = random.random()
    for iA in range(len(actionDistribution)):
      if randomNumber < pActions[iA]:
        return iA
    return 0