"""
By Linda van der Spaa, 2022
l.f.vanderspaa@tudelft.nl
"""

import numpy as np

from AgentModel import AgentModel
from HumanModel import HumanModel

class IAPreferenceLearner:
  def __init__(self,scenario) -> None:
    self.scenario = scenario
    self.human = HumanModel(scenario, softMaxTemp=25, gamma=0.9)
    self.robot = AgentModel(scenario, softMaxTemp=5, gamma=0.9, actionBound=0.9)
    self.updatePolicy()

    self.nIntentions = scenario.intentionsSet.n

  def resetHumanModel(self):
    self.human.reset(self.scenario)
    self.updatePolicy()

  ## Computing probabilities
  def transitionProbability(self, intention0, intention1):
    intentionBias = 0.95
    return 1. if self.nIntentions == 1 else \
          intentionBias if np.linalg.norm(intention1-intention0) < 1e-9 else \
          (1-intentionBias)/(self.nIntentions-1)

  def observationProbability(self, state, ah, intention):
    stateIdx = self.scenario.getStateIdx(state)
    intentionIdx = self.scenario.getIntentionIdx(intention)
    ahIdx = self.scenario.getActionIdx(stateIdx,ah)

    totalProbability = 0.
    for i in range(self.nIntentions):
      totalProbability += self.human.actionProbabilities[i][stateIdx][ahIdx]
    return self.human.actionProbabilities[intentionIdx][stateIdx][ahIdx]/totalProbability

  def beliefTransition(self, s,b,ar,ah,o):
    (sNew,_) = self.scenario.stateTransition(s, ar, ah)
    if np.max(np.abs(o-sNew)) > 1e-9:
      print("State transition does not match observation")
      return []
      
    bNew = [0 for _ in range(self.nIntentions)]
    norm = 0
    for i in range(self.nIntentions):
      iInitial = self.scenario.getIntention(i)
      for j in range(self.nIntentions):
        iNew = self.scenario.getIntention(j)
        p = self.observationProbability(s,ah,iNew) * \
            self.transitionProbability(iInitial,iNew)*b[i]
        bNew[j] += p
        norm += p
    
    if not norm:
      return []
    
    for i in range(self.nIntentions):
      bNew[i] /= norm
    return bNew

  def updateModel(self, trace, intentionIdx):
    prunedTrace = self.scenario.pruneTrajectoryTrace(trace, intentionIdx)
    self.human.updateModel(prunedTrace, self.robot.actionProbabilities)
    self.updatePolicy()

  def updatePolicy(self):
    (P, R) = self.scenario.getMatricesTR('robot', self.human.actionProbabilities, self.human.reward)
    self.robot.updatePolicyPerIntention(P, R)

  def getAction(self, stateIdx, belief):
    actionIdx = self.robot.getActionIdx(stateIdx, belief)
    return self.scenario.getAction(stateIdx, actionIdx)