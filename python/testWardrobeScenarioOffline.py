"""
By Linda van der Spaa, 2022
l.f.vanderspaa@tudelft.nl
"""

import time
import numpy as np

from WardrobeScenarioModel import WardrobeScenario
from IAPreferenceLearner import IAPreferenceLearner
#%%

class AgentTester():
#%%
  def __init__(self, agentType:str) -> None:
    self.scenario = WardrobeScenario(4)

    if agentType not in ['learner', 'imitator']:
      print("Coose valid agent type ('learner' or 'imitator'). Exiting.")
      return
    self.robotPolicy = agentType

#%%
  def initLearning(self) -> None:
    self.learner = IAPreferenceLearner(self.scenario) if self.robotPolicy == 'learner' else None
    self.trajectoryTraces = []
    self.intentionsToTraces = []

#%%
  def learnOnTask(self, startSupportIdx:int, userInputModel=None, debugLevel:int=0):
    # Set/check user input function
    if userInputModel is None:
      def reqNextState():
        print("Please enter next state (int):")
        return int(input())
      userInputModel = lambda s,ar: reqNextState()
      debugLevel = max(debugLevel,1)

    # Initialize
    self.scenario.setStartState(startSupportIdx)
    state, belief = self.scenario.getStartStateBelief()
    stateTrace = []
    beliefTrace = [belief]
    robotActionTrace = []
    humanActionTrace = []
    # Run task
    while not self.scenario.isSupportState(state) or not len(stateTrace):
      stateIdx = self.scenario.getStateIdx(state)
      if self.robotPolicy == 'learner':
        robotAction = self.learner.getAction(stateIdx, beliefTrace[-1])
      elif self.robotPolicy == 'imitator':
        robotAction = self.scenario.getImitatedAction(state)
      else:
        robotAction = self.scenario.getPassiveAction(state)
      actionIdx = self.scenario.getActionIdx(stateIdx,robotAction)
      if debugLevel >= 1:
        print("In current state, {}".format(stateIdx))
        if debugLevel >= 2 and self.robotPolicy == 'learner':
          print("Robot action probabilities are: {}".format(self.learner.robot.getActionDistribution(stateIdx, beliefTrace[-1])))
        print('Robot chooses action: "{}"'.format(robotAction))
        time.sleep(1)
      nextStateIdx = userInputModel(stateIdx, robotAction)
      nextState = self.scenario.statesSet.data[nextStateIdx]

      inferredStates, robotActions, humanActions = self.scenario.inferHumanAction(state, nextState, robotAction)
      stateTrace += inferredStates[:-1]
      robotActionTrace += robotActions
      humanActionTrace += humanActions
      if self.robotPolicy == 'learner':
        for i in range(len(inferredStates)-1):
          beliefTrace.append(self.learner.beliefTransition(inferredStates[i], beliefTrace[-1], robotActions[i], humanActions[i], inferredStates[i+1]))
        if debugLevel >= 1:
          print('Intention belief ' + str(beliefTrace[-1]))
      state = nextState
    stateTrace.append(state)

    self.trajectoryTraces += [[[stateTrace[i], beliefTrace[i], robotActionTrace[i], humanActionTrace[i], stateTrace[i+1]] for i in range(len(robotActionTrace))]]
    self.intentionsToTraces += [self.scenario.getIntentionIdx(stateTrace[-1][:-1])]

    if self.robotPolicy == 'imitator':
      for i in range(len(humanActionTrace)):
        self.scenario.updateActionImitationList(stateTrace[i], humanActionTrace[i])
    elif self.robotPolicy == 'learner':
      self.learner.updateModel(self.trajectoryTraces[-1:], self.intentionsToTraces[-1:])
