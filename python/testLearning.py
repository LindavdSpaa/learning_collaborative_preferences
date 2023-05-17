import itertools, random
import numpy as np
from datetime import date
from humanPreferenceSet import prefSet
from testWardrobeScenarioOffline import AgentTester

# %%
learner = AgentTester('learner')

n_pref = prefSet.shape[-1]**2
n_iter = 24
n_intentions = learner.scenario.intentionsSet.n
n_stateActions = sum([len(sa) for sa in learner.scenario.actionsSet])

# %%
taskOrder = [1, 0, 1, 0, 2, 1, 2, 0, 1]
n_iter = len(taskOrder)-1

# %%
prefSelection = [(1,0), (5,1), (5,0), (1,4), (5,5), (0,0), (2,0), (1,2), (3,1)]
n_pref = len(prefSelection)

# %%
N = 100
learnedActionProbabilities = np.zeros((n_pref,N,n_iter+1,n_intentions,n_stateActions))
for p in range(n_pref):
  print("Preference: {}".format(p))
  for j in range(N):
    if not j%10:
      print(j)
    p_ij = (p//6, p%6) if n_pref == 36 else prefSelection[p]

    # Init learner and store initial policy
    learner.initLearning()
    for i in range(n_intentions):
      intention = np.zeros(n_intentions)
      intention[i] = 1
      learnedActionProbabilities[p,j,0,i] = list(itertools.chain(*[learner.learner.robot.getActionDistribution(s,intention) for s in range(learner.scenario.statesSet.n)]))

    for k in range(n_iter):
      # print("Iteration: {}".format(k))
      rand2x = (random.randint(0, 2), random.randint(1, 2))
      goalSupport = rand2x[0] #taskOrder[k+1] #k%2 #
      startSupport = sum(rand2x)%3 #taskOrder[k] #(k+1)%2 #

      def getNextState(stateIdx:int, robotAction:str) -> int:
        robotGraspDeltaStateIdx = 2 - 4*(stateIdx%4//2) if stateIdx<12 and robotAction=='(un)grasp' else 0
        humanGraspDeltaStateIdx = 0 if stateIdx >= 12 else -(stateIdx%2) + (0 if stateIdx//4==goalSupport else 1)
        prefSeq = prefSet[startSupport,goalSupport -(1 if startSupport<goalSupport else 0),p_ij[min(goalSupport,1)]]
        nextStateIdx = stateIdx + robotGraspDeltaStateIdx + humanGraspDeltaStateIdx if stateIdx<12 and stateIdx not in prefSeq[:(-1 if prefSeq[-2]>=12 else -2)] else \
                        prefSeq[np.where(prefSeq==stateIdx)[0][0]+1]
        return nextStateIdx

      learner.learnOnTask(startSupport, userInputModel=getNextState)

      for i in range(n_intentions):
        intention = np.zeros(n_intentions)
        intention[i] = 1
        learnedActionProbabilities[p,j,k+1,i] = list(itertools.chain(*[learner.learner.robot.getActionDistribution(s,intention) for s in range(learner.scenario.statesSet.n)]))


#%%
np.savez('{}-learned-policies-p{}x{}x{}'.format(date.today(),n_pref,n_iter,N), learnedActionProbabilities=learnedActionProbabilities)
# %%