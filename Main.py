# -*- coding: utf-8 -*-
"""
#Project [Gait Classfication][20191110]

#@author: lukious
"""
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

np.random.seed(1)

row_data = pd.read_csv("./190619_RJR_30Hz_3kph.csv")
train_data = pd.DataFrame(row_data,columns = ["L_sensor1","L_seonsor2","L_sensor3","L_sensor4","R_sensor1","R_sensor2","R_sensor3","R_sensor4","L_accX","L_accY","L_accZ","L_gyroX","L_gyroY","L_gyroZ","R_accX","R_accY","R_accZ","R_gyroX","R_gyroY","R_gyroZ","COP_Left","COP_Right","COP_Front","COP_Back","COP_LeftFront","COP_LeftBack","COP_RightFront","COP_RightBack"])
row_data = pd.read_csv("./190619_RJR_30Hz_5kph.csv")
appending_list= pd.DataFrame(row_data,columns = ["L_sensor1","L_seonsor2","L_sensor3","L_sensor4","R_sensor1","R_sensor2","R_sensor3","R_sensor4","L_accX","L_accY","L_accZ","L_gyroX","L_gyroY","L_gyroZ","R_accX","R_accY","R_accZ","R_gyroX","R_gyroY","R_gyroZ","COP_Left","COP_Right","COP_Front","COP_Back","COP_LeftFront","COP_LeftBack","COP_RightFront","COP_RightBack"])

train_data.append(appending_list)
print(train_data)

# train_data.plot.line()


L_sensors = train_data[["L_sensor1","L_seonsor2","L_sensor3","L_sensor4"]]
L_accs = train_data[["L_accX","L_accY","L_accZ"]]
L_gyros = train_data[["L_gyroX","L_gyroY","L_gyroZ"]]

R_sensors = train_data[["R_sensor1","R_sensor2","R_sensor3","R_sensor4"]]
R_accs = train_data[["R_accX","R_accY","R_accZ"]]
R_gyros = train_data[["R_gyroX","R_gyroY","R_gyroZ"]]

COPs = train_data[["COP_Left","COP_Right","COP_Front","COP_Back","COP_LeftFront","COP_LeftBack","COP_RightFront","COP_RightBack"]]

Sensors = train_data[["L_sensor1","L_seonsor2","L_sensor3","L_sensor4","R_sensor1","R_sensor2","R_sensor3","R_sensor4"]]
Gyros = train_data[["L_gyroX","L_gyroY","L_gyroZ","R_gyroX","R_gyroY","R_gyroZ"]]
without_COP_all = train_data[["L_sensor1","L_seonsor2","L_sensor3","L_sensor4","R_sensor1","R_sensor2","R_sensor3","R_sensor4","L_accX","L_accY","L_accZ","L_gyroX","L_gyroY","L_gyroZ","R_accX","R_accY","R_accZ","R_gyroX","R_gyroY","R_gyroZ"]]
Accs = train_data[["L_accX","L_accY","L_accZ","R_accX","R_accY","R_accZ"]]

X_train, X_test, y_train, y_test = train_test_split(
    input_x,
    input_y,
    test_size=0.3, random_state=0)

X = tf.placeholder(dtype=tf.float32, shape=[None, num_features])
Y = tf.placeholder(dtype=tf.int64, shape=[None])

model = TSNE(learning_rate = 100)
transformed_a = model.fit_transform(Accs)
xs = transformed_a[:,0]
ys = transformed_a[:,1]
plt.scatter(xs,ys)
plt.show()


