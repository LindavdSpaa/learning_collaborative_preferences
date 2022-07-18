"""
By Linda van der Spaa, 2022
l.f.vanderspaa@tudelft.nl
"""

import time
import numpy as np
from irl import maxent
from AgentModel import AgentModel

class HumanModel(AgentModel):
  def __init__(self, scenario, softMaxTemp, gamma) -> None:
    scenario.setStartState()
    self.featureMatrix = scenario.getFeatureMatrix()
    self.getTransitionMatrix = lambda piR: scenario.getMatricesTR('human', piR)
    super().__init__(scenario, softMaxTemp, gamma)

  def reset(self, scenario):
    super().reset(scenario)
    self.transitionMatrix = self.getTransitionMatrix(self.actionProbabilities)
    self.theta = -1*scenario.getStateFeatures(np.append(scenario.intentionsSet.data[0],0), scenario.intentionsSet.data[1]) + 0.1*scenario.getStateFeatures(np.append(scenario.intentionsSet.data[0],0), scenario.intentionsSet.data[0])
    self.updateRewardsPolicy()

  def updateModel(self, prunedTrajectory, robotPolicy, debugLevel=0):
    self.transitionMatrix = self.getTransitionMatrix(robotPolicy)

    if debugLevel >= 1:
      print('Computing IRL solution...')
      tStart = time.time()
    self.theta = maxent.irl(self.featureMatrix, self.nMaxActions, self.gamma,
                            self.transitionMatrix, prunedTrajectory,
                            self.theta, 0.1, 1)
    if debugLevel >= 1:
      dt = time.time() - tStart
      print('Found solution in {}s\n'.format(dt))
    if debugLevel >= 2:
      print('Theta = ', self.theta, '\n')

    self.updateRewardsPolicy()

  def updateRewardsPolicy(self):
    rewards = self.featureMatrix.dot(self.theta)
    # store as matrix of shape (nIntentions, nStates)
    self.reward = np.reshape(rewards, (len(self.actionProbabilities), len(self.actionProbabilities[0])))

    # Update policy
    self.updatePolicy(self.transitionMatrix, rewards)