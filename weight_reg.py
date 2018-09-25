from __future__ import print_function

import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import scipy.io
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

mat = scipy.io.loadmat("C:\\Users\Rui\.PyCharm2018.2\config\scratches\mnistdnn_300_sgd.mat")
weights_log_t_dnn=mat["weights_log_t"]

tempstd=np.std(weights_log_t_dnn,1)
#idx=np.where(tempstd>np.mean(tempstd))
idx=np.where(tempstd>0.01)
weights_log_t_del_dnn=weights_log_t_dnn[idx]

#weights_log_t_del_dnn = np.delete(weights_log_t_dnn, slice(0, 660672), 0)
#weights_log_t_del_dnn=normalize(weights_log_t_del_dnn, norm='l2')
weights_log_t_del_dnn=weights_log_t_del_dnn.transpose()
for x in range(275):
    temp_m = weights_log_t_del_dnn[x:x + 25]
    if x == 0:
        sliced_weights_log_dnn = temp_m.transpose()
    else:
        sliced_weights_log_dnn = np.append(sliced_weights_log_dnn, temp_m.transpose(), axis=0)


'''
mat = scipy.io.loadmat("C:\\Users\Rui\.PyCharm2018.2\config\scratches\mnistcnn_300_sgd.mat")
weights_log_t_cnn=mat["weights_log_t"]
weights_log_t_del = np.delete(weights_log_t_cnn, slice(0, 1199448), 0)
weights_log_t_del_cnn = np.delete(weights_log_t_cnn, slice(0, 1199548), 0)
#weights_log_t_del_cnn=normalize(weights_log_t_del_cnn, norm='l2')
weights_log_t_del_cnn=weights_log_t_del_cnn.transpose()
for x in range(285):
    temp_m = weights_log_t_del_cnn[x:x + 10]
    if x == 0:
        sliced_weights_log_cnn = temp_m.transpose()
    else:
        sliced_weights_log_cnn = np.append(sliced_weights_log_cnn, temp_m.transpose(), axis=0)

mati = scipy.io.loadmat("C:\\Users\Rui\.PyCharm2018.2\config\scratches\imdbcnn_300_sgd.mat")
weights_log_t_imdbcnn=mati["weights_log_t"]
weights_log_t_del_imdbcnn = np.delete(weights_log_t_imdbcnn, slice(0, 99900), 0)
#weights_log_t_del_imdbcnn=normalize(weights_log_t_del_imdbcnn, norm='l2')
weights_log_t_del_imdbcnn=weights_log_t_del_imdbcnn.transpose()
for x in range(285):
    temp_m = weights_log_t_del_imdbcnn[x:x + 10]
    if x == 0:
        sliced_weights_log_imdbcnn = temp_m.transpose()
    else:
        sliced_weights_log_imdbcnn = np.append(sliced_weights_log_imdbcnn, temp_m.transpose(), axis=0)

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
#sliced_weights_log_merge=np.append(sliced_weights_log_cnn, sliced_weights_log_imdbcnn, axis=0)
#sliced_weights_log_merge=np.append(sliced_weights_log_merge, sliced_weights_log_dnn, axis=0)
#sliced_weights_log_merge=np.append(sliced_weights_log_merge, sliced_weights_log_cifarcnn, axis=0)
#sliced_weights_log_merge=normalize(sliced_weights_log_merge, norm='l2')
#liced_weights_log_merge=sliced_weights_log_merge*1000
#sliced_weights_log_merge=sliced_weights_log_dnn

np.random.seed(1)
np.random.shuffle(sliced_weights_log_merge)

#tempstd=np.std(sliced_weights_log_merge,1)
#idx=np.where(tempstd>np.mean(tempstd))
#idx=np.where(tempstd>0.01)
#temp_mat=sliced_weights_log_merge[idx]

temp_mat=sliced_weights_log_merge

temp_l= temp_mat.shape[0]

X=temp_mat[0:temp_l,0:5]
Y=temp_mat[0:temp_l,24]

x_train, x_test, y_train, y_test = train_test_split(
X, Y, test_size=0.33, random_state=42)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

regr = KNeighborsRegressor()
regr.fit(x_train, y_train)

print(regr.score(x_test,y_test))

import pickle
filename = 'pred25_finalized_reg_model.sav'
pickle.dump(regr, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
print(loaded_model.score(x_test,y_test))
