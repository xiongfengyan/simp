import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utlib import input_data
from model_ae import gaemodel

import geopandas as gpd
from geopandas import GeoSeries
from shapely.geometry import LineString

import sys
import time
import math


early_stopping =5000

path = 'data'+'/landuse_line_one.shp'
layers = [14, 15, 16]
num_layers = len(layers)
num_orders = 3
features, all_supports, all_marks, max_x_list, max_y_list, min_x_list, min_y_list, start_x_list, start_y_list, end_x_list,\
end_y_list, pad = input_data(path, num_layers, num_orders)

print(all_marks)
print(len(all_marks))
print(len(all_marks[0][0]))

#exit()

placeholder = {
    "L" : [[tf.compat.v1.sparse_placeholder(tf.float32) for _ in range(num_orders+1)] for _ in range(num_layers)],
    "feature" : tf.compat.v1.placeholder(tf.float32,shape=(None, 2))
}

out_shp = []
epochs = 3000
all_cost = []
time_start=time.time()

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
with tf.compat.v1.Session(config=config) as sess:
    for i in range(len(features)):
        supports, marks = all_supports[i], all_marks[i]
        area = 0
        
        le = len(features[i])
        dis = math.sqrt((features[i][0][0]-features[i][le-1][0]) * (features[i][0][0]-features[i][le-1][0])
                   + (features[i][0][1]-features[i][le-1][1]) * (features[i][0][1]-features[i][le-1][1]))
        
        dis_total = 0.0
        for j in range(le-1):
            dis_total += math.sqrt((features[i][j][0]-features[i][j+1][0]) * (features[i][j][0]-features[i][j+1][0])
                                   + (features[i][j][1]-features[i][j+1][1]) * (features[i][j][1]-features[i][j+1][1]))
        for j in range(le-1):
            #print(features[i][j])
            area += features[i][j][0]*features[i][j+1][1] - features[i][j+1][0]*features[i][j][1]
        area += features[i][le-1][0]*features[i][0][1] - features[i][0][0]*features[i][le-1][1]   #大家都没除2
        print(marks)
        
        model = gaemodel(placeholder, layers, marks, area)
        sess.run(tf.compat.v1.global_variables_initializer())

        cost = []
        shape = list()
        for j in range(epochs):
            shape, loss, l1, l2, l3, l4 = model.train(sess, supports, features[i])
            if j > 1000:
                cost.append(loss)
            if j % 250 == 0:
                print(i, j, loss, l1, l2, l3, l4)
        print('area = ', area)
        all_cost.append(cost)

        out = model.reconstruct(sess, supports, features[i])
        out = out.T
        
        x = list(out[0]*(max_x_list[i] - min_x_list[i])+min_x_list[i])
        y = list(out[1]*(max_y_list[i] - min_y_list[i])+min_y_list[i])
        x[0] = start_x_list[i]
        y[0] = start_y_list[i]

        x[-1] = end_x_list[i]
        y[-1] = end_y_list[i]
        plt.plot(x, y)
        line_t = LineString(list(zip(x, y)))
        out_shp.append(line_t)

time_end=time.time()
print('totally time', round(time_end-time_start, 3))


#g = GeoSeries(out_shp)
#g.to_file('data/building_50073.shp')
#plt.title('graph_ae_dis'+str(epochs))

plt.show()

'''
for i in all_cost:
    plt.plot(i)
    plt.show()
    '''