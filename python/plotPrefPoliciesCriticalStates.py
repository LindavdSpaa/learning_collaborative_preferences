import sys
import numpy as np
import matplotlib.pyplot as plt
from WardrobeScenarioModel import WardrobeScenario
from humanPreferenceSet import prefSet

# %%
scenario = WardrobeScenario(None)

filename = str(sys.argv[1])
data = np.load(filename)
learnedActionProbabilities = data['learnedActionProbabilities']

# %%
stateActionIndices = np.cumsum([len(sa) for sa in scenario.actionsSet])

criticalStates = np.array([[19, 17, 18, 15],
                           [12, 16, 13, 14]], dtype=int)

# %%
plt.figure(figsize=(20,20))
for p in range(len(learnedActionProbabilities)):
  p_ij = (p//6, p%6)
  ax = plt.subplot(6,6,p+1)
  x_max = learnedActionProbabilities.shape[1]-1
  ax.plot([0,x_max],[1,1],'k-.')
  ax.set_xlim([0,x_max])
  for i in range(len(criticalStates)):
    plt.gca().set_prop_cycle(None)
    prefSeq = prefSet[(i+1)%2,0,p_ij[i]]
    for s in criticalStates[i]:
      if s in prefSeq:
        idx0 = stateActionIndices[s-1]
        idx1 = stateActionIndices[s]
        finalProbabilities = learnedActionProbabilities[p,-1,i,idx0:idx1]
        idxMax = np.argmax(finalProbabilities)
        idxNext = np.argmax(finalProbabilities[list(range(idxMax))+list(range(idxMax+1,idx1-idx0))])
        if idxNext >= idxMax:
          idxNext += 1
        ax.plot(learnedActionProbabilities[p,:,i,idx0+idxMax]/learnedActionProbabilities[p,:,i,idx0+idxNext],('-' if i else '--'), label=str(s))
  ax.legend()
plt.show()

# %%
p = 30
states = np.array([[19, 15, 14, 12],[12, 16, 17, 19]], dtype=int)
for i in range(len(states)):
  for s in range(len(states[i])):
    sIdx = states[i,s]
    idx0 = stateActionIndices[sIdx-1]
    idx1 = stateActionIndices[sIdx]
    finalProbabilities = learnedActionProbabilities[p,-1,i,idx0:idx1]
    idxMax = np.argmax(finalProbabilities)
    idxNext = np.argmax(finalProbabilities[list(range(idxMax))+list(range(idxMax+1,idx1-idx0))])
    if idxNext >= idxMax:
      idxNext += 1
    print('First and second actions in state {} (going to g{}): '.format(sIdx,i) +\
          scenario.actionsSet[sIdx][idxMax]+', '+scenario.actionsSet[sIdx][idxNext])
# %%
