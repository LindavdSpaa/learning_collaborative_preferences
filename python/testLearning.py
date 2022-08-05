import itertools
import numpy as np
from datetime import date
from humanPreferenceSet import prefSet
from testWardrobeScenarioOffline import AgentTester

# %%
learner = AgentTester('learner')

n_pref = prefSet.shape[-1]**2
n_iter = 20
n_intentions = learner.scenario.intentionsSet.n
n_stateActions = sum([len(sa) for sa in learner.scenario.actionsSet])

# %%
learnedActionProbabilities = np.zeros((n_pref,n_iter+1,n_intentions,n_stateActions))
for p in range(n_pref):
  print("Preference: {}".format(p))
  p_ij = (p//6, p%6)

  # Init learner and store initial policy
  learner.initLearning()
  for i in range(n_intentions):
    intention = np.zeros(n_intentions)
    intention[i] = 1
    learnedActionProbabilities[p,0,i] = list(itertools.chain(*[learner.learner.robot.getActionDistribution(s,intention) for s in range(learner.scenario.statesSet.n)]))

  for k in range(n_iter):
    print("Iteration: {}".format(k))
    goalSupport = k%2
    startSupport = (k+1)%2

    def getNextState(stateIdx:int, robotAction:str) -> int:
      robotGraspDeltaStateIdx = 2 - 4*(stateIdx%4//2) if stateIdx<12 and robotAction=='(un)grasp' else 0
      humanGraspDeltaStateIdx = 0 if stateIdx >= 12 else -(stateIdx%2) + (0 if stateIdx//4==goalSupport else 1)
      prefSeq = prefSet[startSupport,goalSupport -(1 if startSupport<goalSupport else 0),p_ij[min(goalSupport,1)]]
      nextStateIdx = stateIdx + robotGraspDeltaStateIdx + humanGraspDeltaStateIdx if stateIdx<12 and stateIdx not in prefSeq[:-1] else \
                      prefSeq[np.where(prefSeq==stateIdx)[0][0]+1]
      return nextStateIdx

    learner.learnOnTask(startSupport, userInputModel=getNextState)

    for i in range(n_intentions):
      intention = np.zeros(n_intentions)
      intention[i] = 1
      learnedActionProbabilities[p,k+1,i] = list(itertools.chain(*[learner.learner.robot.getActionDistribution(s,intention) for s in range(learner.scenario.statesSet.n)]))


#%%
np.savez('{}-learned-policies-p{}x{}'.format(date.today(),n_pref,n_iter), learnedActionProbabilities=learnedActionProbabilities)
# %%