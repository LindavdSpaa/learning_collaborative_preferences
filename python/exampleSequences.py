import numpy as np
from WardrobeScenarioModel import WardrobeScenario

scenario = WardrobeScenario(4)

sequence0 = [7, 19, 17, 13, 12, 3]
stateTrace0 = np.array([scenario.statesSet.data[s] for s in sequence0])
actionHTrace0 = ['take off support','move down to 0','move over to 0',
                  'rotate','put on support']

sequence1 = [3, 12, 16, 17, 19, 7]
stateTrace1 = np.array([scenario.statesSet.data[s] for s in sequence1])
actionHTrace1 = ['take off support','move over to 1','rotate',
                  'move up to 1','put on support']


sequence0N = [19, 17, 13, 12, 3]
stateTrace0N = np.array([scenario.statesSet.data[s] for s in sequence0N])
actionHTrace0N = ['move over to 2','rotate','move over to 2','move up to 1']

sequence1N = [12, 16, 17, 19, 7]
stateTrace1N = np.array([scenario.statesSet.data[s] for s in sequence1N])
actionHTrace1N = ['move over to 2','move over to 0','move over to 2','move over to 0']