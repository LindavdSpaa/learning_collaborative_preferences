"""
By Linda van der Spaa, 2022
l.f.vanderspaa@tudelft.nl
"""

import time
import numpy as np

from WardrobeScenarioModel import WardrobeScenario
from IAPreferenceLearner import IAPreferenceLearner
#%%
#%%
scenario = WardrobeScenario(3)
robotPolicy = 'learner'  # 'learner', 'imitator'
if robotPolicy == 'learner':
  learner = IAPreferenceLearner(scenario)

#%%
trajectoryTraces = []
intentionsToTraces = []

#%%
print("Enter start support (int):")
startSupport = int(input())
scenario.setStartState(startSupport)
#%
state, belief = scenario.getStartStateBelief()
stateTrace = []
beliefTrace = [belief]
robotActionTrace = []
humanActionTrace = []
while not scenario.isSupportState(state) or not len(stateTrace):
  stateIdx = scenario.getStateIdx(state)
  if robotPolicy == 'learner':
    robotAction = learner.getAction(stateIdx, beliefTrace[-1])
  elif robotPolicy == 'imitator':
    robotAction = scenario.getImitatedAction(state)
  else:
    robotAction = scenario.getPassiveAction(state)
  actionIdx = scenario.getActionIdx(stateIdx,robotAction)
  print("In current state, {}".format(stateIdx))
  if robotPolicy == 'learner':
    print("Robot action probabilities are: {}".format(learner.robot.getActionDistribution(stateIdx, beliefTrace[-1])))
  print('Robot chooses action: "{}"'.format(robotAction))
  time.sleep(1)
  print("Please enter next state (int):")
  nextStateIdx = int(input())
  nextState = scenario.statesSet.data[nextStateIdx]

  inferredStates, robotActions, humanActions = scenario.inferHumanAction(state, nextState, robotAction)
  stateTrace += inferredStates[:-1]
  robotActionTrace += robotActions
  humanActionTrace += humanActions
  if robotPolicy == 'learner':
    for i in range(len(inferredStates)-1):
      beliefTrace.append(learner.beliefTransition(inferredStates[i], beliefTrace[-1], robotActions[i], humanActions[i], inferredStates[i+1]))
    print('Intention belief ' + str(beliefTrace[-1]))
  state = nextState
stateTrace.append(state)

trajectoryTraces += [[[stateTrace[i], beliefTrace[i], robotActionTrace[i], humanActionTrace[i], stateTrace[i+1]] for i in range(len(robotActionTrace))]]
intentionsToTraces += [scenario.getIntentionIdx(stateTrace[-1][:-1])]
#%%
if robotPolicy == 'imitator':
  for i in range(len(humanActionTrace)):
    scenario.updateActionImitationList(stateTrace[i], humanActionTrace[i])
elif robotPolicy == 'learner':
  learner.updateModel(trajectoryTraces[-1:], intentionsToTraces[-1:])

#%%
