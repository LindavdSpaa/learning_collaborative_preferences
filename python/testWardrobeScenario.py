import rospy, time, os, itertools
from datetime import datetime
from std_msgs.msg import Int16, String
import numpy as np

from WardrobeScenAbstractActionController import WardrobeAbstractActionController
from WardrobeScenarioModel import WardrobeScenario
from IAPreferenceLearner import IAPreferenceLearner
#%%
#%%
rospy.init_node('Linda', anonymous=True)
time.sleep(5)
statePub = rospy.Publisher('/scenario_abstract_state', Int16, queue_size=10)
actionPub = rospy.Publisher('/scenario_abstract_action', String, queue_size=10)
time.sleep(5)

#%%
actionController = WardrobeAbstractActionController(True)
scenario = WardrobeScenario(4)
robotPolicy = 'learner'  # 'learner', 'imitator'
if robotPolicy == 'learner':
  learner = IAPreferenceLearner(scenario)

  n_intentions = scenario.intentionsSet.n
  n_stateActions = sum([len(sa) for sa in scenario.actionsSet])
  learnedPolicy = np.zeros((30,n_intentions,n_stateActions))

  for i in range(n_intentions):
    intention = np.zeros(n_intentions)
    intention[i] = 1
    learnedPolicy[0,i] = list(itertools.chain(*[learner.robot.getActionDistribution(s,intention) for s in range(scenario.statesSet.n)]))
elif robotPolicy == 'imitator':
  learnedPolicy = []

trajectoryTraces = []
intentionsToTraces = []
beliefsTraces = []
k = 0

#%% Start with an open gripper
actionController.grasp(False)

#%%
actionController.goToGoal(1)






#%%
state, stateIdx = actionController.goToClosestGoal()
scenario.setStartState(stateIdx//4)
statePub.publish(stateIdx)
#%
time.sleep(1)
stateTrace = []
beliefTrace = [scenario.getStartStateBelief()[1]]
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
  actionPub.publish(robotAction)
  nextState, nextStateIdx = actionController.doAction(stateIdx, actionIdx)
  statePub.publish(nextStateIdx)
  print('Arrived at state {}'.format(nextStateIdx))

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

##%%
if robotPolicy == 'imitator':
  for i in range(len(humanActionTrace)):
    scenario.updateActionImitationList(stateTrace[i], humanActionTrace[i])
  trajectoryTraces += [[[stateTrace[i], robotActionTrace[i], humanActionTrace[i], stateTrace[i+1]] for i in range(len(robotActionTrace))]]
  learnedPolicy += [scenario.stateActionImitationList]
elif robotPolicy == 'learner':
  trajectoryTraces += [[[stateTrace[i], beliefTrace[i], robotActionTrace[i], humanActionTrace[i], stateTrace[i+1]] for i in range(len(robotActionTrace))]]
  intentionsToTraces += [scenario.getIntentionIdx(stateTrace[-1][:-1])]
  beliefsTraces += [beliefTrace]
  learner.updateModel(trajectoryTraces[-1:], intentionsToTraces[-1:])

  k += 1
  for i in range(n_intentions):
    intention = np.zeros(n_intentions)
    intention[i] = 1
    learnedPolicy[k,i] = list(itertools.chain(*[learner.robot.getActionDistribution(s,intention) for s in range(scenario.statesSet.n)]))


# %%
np.savez(datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+'-learned-policy-'+robotPolicy,
          trajectories=np.array(trajectoryTraces, dtype=object),
          intentions=np.array(intentionsToTraces, dtype=object), 
          beliefs=np.array(beliefsTraces, dtype=object), 
          learnedPolicy=np.array(learnedPolicy, dtype=object))
# %%
