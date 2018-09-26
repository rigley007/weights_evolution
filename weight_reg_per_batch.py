from __future__ import print_function

import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import scipy.io
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import pickle

filename = 'mnistdnn_5_sgd_batch_std02.pckl'
weights_log_t_dnn = pickle.load(open(filename, 'rb'))
'''
tempstd=np.std(weights_log_t_dnn,1)
idx=np.where(tempstd>1.5*np.mean(tempstd))
weights_log_t_del_dnn=weights_log_t_dnn[idx]
'''
weights_log_t_del_dnn=weights_log_t_dnn

weights_log_t_del_dnn=weights_log_t_del_dnn.transpose()

for x in range(2330):
    print(x)
    temp_m = weights_log_t_del_dnn[x:x + 15]
    if x == 0:
        sliced_weights_log_dnn = temp_m.transpose()
    else:
        sliced_weights_log_dnn = np.append(sliced_weights_log_dnn, temp_m.transpose(), axis=0)



'''
mat = scipy.io.loadmat("C:\\Users\Rui\.PyCharm2018.2\config\scratches\cifarcnn_300_sgd.mat")
weights_log_t_cifarcnn=mat["weights_log_t"]

tempstd=np.std(weights_log_t_cifarcnn,1)
#idx=np.where(tempstd>np.mean(tempstd))
idx=np.where(tempstd>0.02)
weights_log_t_del_cifarcnn=weights_log_t_cifarcnn[idx]
#weights_log_t_del_cifarcnn = np.delete(weights_log_t_cifarcnn, slice(0, 1240044), 0)
#weights_log_t_del_cifarcnn=normalize(weights_log_t_del_cifarcnn, norm='l2')
weights_log_t_del_cifarcnn=weights_log_t_del_cifarcnn.transpose()
for x in range(275):
    temp_m = weights_log_t_del_cifarcnn[x:x + 25]
    if x == 0:
        sliced_weights_log_cifarcnn = temp_m.transpose()
    else:
        sliced_weights_log_cifarcnn = np.append(sliced_weights_log_cifarcnn, temp_m.transpose(), axis=0)



sliced_weights_log_merge=sliced_weights_log_dnn
'''

np.random.seed(1)
np.random.shuffle(sliced_weights_log_dnn)

#tempstd=np.std(sliced_weights_log_merge,1)
#idx=np.where(tempstd>np.mean(tempstd))
#idx=np.where(tempstd>0.01)
#temp_mat=sliced_weights_log_merge[idx]



temp_l= sliced_weights_log_dnn.shape[0]

X=sliced_weights_log_dnn[0:temp_l,0:5]
Y=sliced_weights_log_dnn[0:temp_l,14]

x_train, x_test, y_train, y_test = train_test_split(
X, Y, test_size=0.33, random_state=42)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

regr = KNeighborsRegressor()
regr.fit(x_train, y_train)

print(regr.score(x_test,y_test))


filename = 'pred15_batch_reg_model_std02.sav'
pickle.dump(regr, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
print(loaded_model.score(x_test,y_test))
