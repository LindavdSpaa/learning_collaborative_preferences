"""
Collection of joint configurations to states set, for generating a GP for approximate IK.

By Linda van der Spaa, 2022
l.f.vanderspaa@tudelft.nl
"""

import numpy as np
import quaternion as Q
from WardrobeScenarioModel import WardrobeScenario

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
import pickle, rosbag

# %% Models
scenario = WardrobeScenario(1)

#%% Get orientations and positions at scenario states
ori0 = scenario.statesSet.data[0,3:7]; qo0=Q.from_float_array(ori0)
ori1 = scenario.statesSet.data[4,3:7]; qo1=Q.from_float_array(ori1)

p0,_= scenario.getRobotPoseWrtObject(scenario.statesSet.data[0,:3], qo0)
p0O,_= scenario.getRobotPoseWrtObject(scenario.statesSet.data[12,:3], qo0)
# p0Or,_= scenario.getRobotPoseWrtObject(scenario.statesSet.data[12,:3], qo1)     # Unreachable!
p0Oh1,_= scenario.getRobotPoseWrtObject(scenario.statesSet.data[14,:3], qo0)
# p0Oh1r,_= scenario.getRobotPoseWrtObject(scenario.statesSet.data[14,:3], qo1)   # Unreachable!
p1Oh0r,_= scenario.getRobotPoseWrtObject(scenario.statesSet.data[16,:3], qo0)
p1Oh0,_= scenario.getRobotPoseWrtObject(scenario.statesSet.data[16,:3], qo1)
p1Or,_= scenario.getRobotPoseWrtObject(scenario.statesSet.data[18,:3], qo0)
p1O,_= scenario.getRobotPoseWrtObject(scenario.statesSet.data[18,:3], qo1)
p1,_= scenario.getRobotPoseWrtObject(scenario.statesSet.data[4,:3], qo1)
p2Oh0r,_= scenario.getRobotPoseWrtObject(scenario.statesSet.data[20,:3], qo0)
p2Oh0,_= scenario.getRobotPoseWrtObject(scenario.statesSet.data[20,:3], qo1)
p2Or,_= scenario.getRobotPoseWrtObject(scenario.statesSet.data[22,:3], qo0)
p2O,_= scenario.getRobotPoseWrtObject(scenario.statesSet.data[22,:3], qo1)
p2,_= scenario.getRobotPoseWrtObject(scenario.statesSet.data[8,:3], qo1)

#%% Corresponding joint configurations
# at p0, ori0
q0 = np.array([1.1415268079806886, -1.2483248437245684, -1.4300334120199183, -1.6705439177018402, 0.09061905684600982, 2.73856184280128, -0.5038090522086905])

# at p0O, ori0
q0O = np.array([0.8265367044105862, -1.1391461967943755, -1.1320318931891404, -2.300839705220457, 0.10719467461652855, 3.085565509359066, -0.2788018345405208])

# at p0O, ori1
q0Or = np.array([1.722698173615993, -1.1690877650726177, -1.6425872275871143, -0.8195465351530717, 0.48508489211400346, 1.0167262973103017, -0.5190183051269255])

# at p0Oh1, ori0
q0Oh1 = np.array([1.081460799505836, -0.8776355276131708, -0.9983404392139573, -1.5146109183629926, -0.1915077234413777, 2.5874915626520867, 0.23734523247383835])

# at p0Oh1, ori1 #(can't quite reach)
q0Oh1r = np.array([1.8684154127763881, -1.0108212398395202, -1.3177029142875936, -0.5166647137848036, 0.5232884919643401, 1.2779183045095792, -0.06196085324445502])


# at p1, ori1
q1 = np.array([2.3074275304783876, -0.7962163708251818, -1.064948932661808, -1.6433088171061014, 0.2705105387187169, 2.672421444318093, -0.13217727560904488])

# at p1O, ori1
q1O = np.array([1.9505505221391977, -0.9531879177177159, -0.8843518009688223, -1.5398176477098122, 0.23092759558997383, 2.2212075154373143, -0.046081957208842894])

# at p1Or, ori0
q1Or = np.array([-0.24173696584324392, -0.9676946288769395, 0.6625065673360341, -1.8920798796436238, -0.10231408782072668, 2.5843300637447593, 1.410461278778322])

# at p1Oh0, ori1
q1Oh0 = np.array([1.5623177409033917, -1.1149333099693703, -1.0367030312887502, -2.2959730967793703, 0.5890977043039525, 2.448727104987945, -0.6651673993309382])

# at p1Oh0r, ori0
q1Oh0r = np.array([0.009225982459229336, -1.1919196317848293, 0.7801501966944222, -2.6019692179055185, -0.627236620980494, 2.89203286002613, 2.121047200932426])


# at p2, ori1
q2 = np.array([1.531330126776188, -0.6752146932996561, -0.2558866264677389, -2.0648350281631753, 0.7589065134997424, 2.8302379402057367, -0.10979331001003526])

# at p2O, ori1
q2O = np.array([1.1125136988361977, -1.0563191812441353, -0.24585424287372154, -2.113565915966038, 0.7932650265128619, 2.3468151072172527, -0.06429818290799726])

# at p2Or, ori0
q2Or = np.array([0.020653125443737593, -1.2084875108522342, 0.6823887392772112, -1.7103974986420125, -0.16930685093868988, 2.0703200252061973, 1.5147658490894502])

# at p2Oh0, ori1
q2Oh0 = np.array([0.8759610418311337, -1.4362994251165655, -0.3585568447046279, -2.8766506368675886, 1.0859477119620498, 2.3416128249568056, -0.6105295661711561])

# at p2Oh0r, ori0
q2Oh0r = np.array([0.6686563257209039, -1.2219655884906828, 0.935226245410027, -2.447135422617747, -0.6444982746895255, 2.0978313230122216, 2.2404718723499655])

#%% Stack into matrices, most neutral pose on top
posOriMat = np.vstack([np.hstack([p1,ori1]),
                       np.hstack([p1O,ori1]),
                       np.hstack([p1Or,ori0]),
                       np.hstack([p1Oh0,ori1]),
                       np.hstack([p1Oh0r,ori0]),
                       np.hstack([p2,ori1]),
                       np.hstack([p2O,ori1]),
                       np.hstack([p2Or,ori0]),
                       np.hstack([p2Oh0,ori1]),
                       np.hstack([p2Oh0r,ori0]),
                       np.hstack([p0,ori0]),
                       np.hstack([p0O,ori0]),
                       np.hstack([p0O,ori1]),
                       np.hstack([p0Oh1,ori0]),
                       np.hstack([p0Oh1,ori1])])
configMat = np.vstack([q1, q1O, q1Or, q1Oh0, q1Oh0r,
                       q2, q2O, q2Or, q2Oh0, q2Oh0r,
                       q0, q0O, q0Or, q0Oh1, q0Oh1r])

#%% Learn IK model
k = C(constant_value=np.sqrt(0.1)) * RBF(1*np.ones(7)) + WhiteKernel(0.01)
gp = GaussianProcessRegressor(kernel=k, alpha=1e-10, n_restarts_optimizer=200)
X_0=posOriMat
Y_0=configMat-configMat[0,:]
gp0 = configMat[0,:]
gp.fit(X_0,Y_0)
gp_kernels_ = gp.kernel_
kernel_params_ = [gp_kernels_.get_params()['k1__k2__length_scale'], gp_kernels_.get_params()['k1']]
print(kernel_params_[0])
noise_var_ = gp.alpha + gp_kernels_.get_params()['k2__noise_level']

# %%
with open('nullspace_gpgp0.pkl','wb') as f:
  pickle.dump(gp,f)
  pickle.dump(gp0,f)
# just to close file
# %%
