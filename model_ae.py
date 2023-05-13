import tensorflow as tf
import numpy as np

from scipy.spatial.distance import euclidean

def _c(ca, i, j, p, q):
    a = tf.gather(ca, i)
    a = tf.gather(a, j)

    cond_1 = tf.cond(tf.greater(ca[i, j], -1), lambda: 0, lambda: 1)
    cond_2 = tf.cond(tf.greater(i, 0), lambda:0 , lambda: 1)
    cond_3 = tf.cond(tf.greater(j, 0), lambda:0 , lambda: 1)
    if cond_1 == 0:
        return ca[i, j]
    elif i == 0 and j == 0:
        ca[i, j] = tf.linalg.norm(tf.gather(p,i)-tf.gather(q, j))
    elif cond_2 == 0 and j == 0:
        ca[i, j] = max(_c(ca, i-1, 0, p, q), tf.linalg.norm(p[i]-q[j]))
    elif i == 0 and cond_3 == 0:
        ca[i, j] = max(_c(ca, 0, j-1, p, q), tf.linalg.norm(p[i]-q[j]))
    elif cond_2 == 0 and cond_3 == 0:
        ca[i, j] = max(
            min(
                _c(ca, i-1, j, p, q),
                _c(ca, i-1, j-1, p, q),
                _c(ca, i, j-1, p, q)
            ),
            tf.linalg.norm(p[i]-q[j])
            )
    else:
        #ca[i, j] = tf.add(ca[i, j], tf.constant(float('inf'), shape=ca[i, j].shape))
        ca[i, j].assign(tf.constant(float('inf')))

    return ca[i, j]

def _frdist(p, q):

    len_p = tf.shape(p)[0]
    len_q = tf.shape(q)[0]
    
    if len_p == 0 or len_q == 0:
        raise ValueError('Input curves are empty.')

    ca = tf.Variable(tf.ones((56, 56), dtype=tf.float32) * -1)
    dist = _c(ca, len_p-1, len_q-1, p, q)
    return dist

def tfDTW(s1, s2):
    #TODO
    return d

class gaemodel(object):
    def __init__(self, placeholder, layers, marks, area):
        self.L = placeholder['L']
        self.feature = placeholder['feature']
        self.placeholder = placeholder
        self.shape = list()
        m =2
        weight_decay = 5e-5
        vars = []
        x = self.feature

        def x_reshape(x_in, size=2, interpolate = False, area = 0):
            loss = 0
            print(size, x_in, '---------------')
            #loss += x_in[ 8][1] * 1000

            loss1 = 0.0     # angle
            loss2 = 0.0     # area
            loss3 = 0.0     # shape
            loss4 = 0.0     # TODO: self-intersection

            #re_area = x_in[0][0]*x_in[1][1] - x_in[1][0]*x_in[0][1]
            re_area = x_in[size-1][0]*x_in[0][1] - x_in[0][0]*x_in[size-1][1]
            for i in range(size-1):
                xd = x_in[i+1][0]-x_in[i][0]
                yd = x_in[i+1][1]-x_in[i][1]

                re_area += x_in[i][0]*x_in[i+1][1] - x_in[i+1][0]*x_in[i][1]
                #loss1 += abs(xd * yd) / (xd*xd + yd*yd + 0.0001)
                loss1 += abs(xd * yd) / (xd*xd + yd*yd + 0.0001)

            #re_area += x_in[size-1][0]*x_in[0][1] - x_in[0][0]*x_in[size-1][1]
            loss2 = abs(re_area - area)

            
            for i in range(1, size-1):
                a = x_in[i-1][0]-x_in[i][0]
                b = x_in[i-1][1]-x_in[i][1]
                c = x_in[i+1][0]-x_in[i][0]
                d = x_in[i+1][1]-x_in[i][1]
                e = tf.sqrt(a*a + b*b + 0.0001)
                f = tf.sqrt(c*c + d*d + 0.0001)

                coss = (a*c+b*d)/(e*f)
                #loss4 += (1+coss)*1.5   #*0.5
                loss4 += tf.cond(coss > 0, lambda: coss, lambda: 0.0)       # To be tested...
                #loss4 += tf.cond(coss > 0, lambda: tf.sqrt((coss*coss)/(1-coss*coss+0.0001)), lambda: 0.0)

            x_ = x_in
            if interpolate:
                x_ = tf.reshape(x_in, shape=(1, tf.shape(x_in)[0], tf.shape(x_in)[1]))
                x_ = tf.image.resize(x_, (1, tf.shape(self.feature)[0]), 0)
                x_ = tf.reshape(x_, shape=(tf.shape(x_)[1], tf.shape(x_)[2]))

            x_ = tf.cast(x_, dtype=tf.float32)
            
            loss3 = (tf.reduce_sum(self.feature*(-1.0)*tf.math.log(x_) + (1-self.feature)*(-1.0)*tf.math.log(1-x_)))

            loss = loss2 + loss3 + loss4       #landuse
            #loss = loss1 + loss2 + loss3 + loss4        #building
            return loss, x_, re_area, loss1, loss2, loss3, loss4

        x, var = GraphConvolution(x, self.L[0], shape=(2, layers[0]), sparse_inputs = False)
        vars.append(var)
        self.x1 = x
        if len(layers) > 1:
            for i in range(len(layers)-1):
                if marks[i+1][0] == 1 or True:
                    m = marks[i+1][2]
                    x = tf.reshape(x, shape=(1, tf.shape(x)[0], tf.shape(x)[1]))
                    x = tf.nn.pool(input=x, window_shape=[m], pooling_type="AVG", padding="SAME",strides=[m])
                    x = tf.reshape(x, shape=(tf.shape(x)[1], tf.shape(x)[2]))

                print(x.shape, '----')
                x, var = GraphConvolution(x, self.L[i+1], shape=(layers[i], layers[i+1]))
                vars.append(var)
                self.shape.append(tf.shape(x))

        self.encoder = x
        self.x2 = x
        self.loss = 0

        interpolate = True
        if interpolate:
            for i in range(0, len(layers)-1):
                if marks[-(i+1)][0] == 1 or True:
                    if i != 0: continue

                    # self.shape.append(tf.shape(x))
                    x = tf.reshape(x, shape=(1, tf.shape(x)[0], tf.shape(x)[1]))
                    # x = tf.keras.layers.UpSampling1D(size=2)(x)
                    x = tf.image.resize(x, (1, marks[-(i+1)][1]), 0)
                    #x = tf.image.resize_images(x, (1, marks[-(1)][1]), 0)
                    x = tf.reshape(x, shape=(tf.shape(x)[1], tf.shape(x)[2]))

                    #x, var = GraphConvolution(x, self.L[-(1)], shape=(layers[-(i+1)], layers[-(i+2)] if i!=len(layers)-1 else 2))    #不上采样
                    x, var = GraphConvolution(x, self.L[-(1)], shape=(layers[-(1)], 2)) 
                    vars.append(var)
                else:
                    x = tf.reshape(x, shape=(1, tf.shape(x)[0], tf.shape(x)[1]))
                    x = tf.image.resize_images(x, (1, marks[-(i+2)][1]), 0)
                    x = tf.reshape(x, shape=(tf.shape(x)[1], tf.shape(x)[2]))
                    if i == 0:
                        x, var = GraphConvolution(x, self.L[-(2)], shape=(layers[-(2)], layers[-(3)]))
                        vars.append(var)
                    if i == 1:
                        x, var = GraphConvolution(x, self.L[-(3)], shape=(layers[-(3)], 2))
                        vars.append(var)
        else: 
            for i in range(0, len(layers)-1):
                if marks[-(i+1)][0] == 1 or True:
                    x, var = GraphConvolution(x, self.L[-(i+1)], shape=(layers[-(i+1)], layers[-(i+2)]))
                    vars.append(var) 

                    x = tf.reshape(x, shape=(1, tf.shape(x)[0], tf.shape(x)[1]))
                    x = tf.image.resize_images(x, (1, marks[-(i+2)][1]), 0)
                    x = tf.reshape(x, shape=(tf.shape(x)[1], tf.shape(x)[2]))
            x, var = GraphConvolution(x, self.L[-len(layers)], shape=(layers[-len(layers)], 2))
            vars.append(var)

        self.x_out = x
        self.x3 = x
        #self.loss += x_reshape(self.x_out)
        loss__, x__, area__, loss1_, loss2_, loss3_, loss4_ = x_reshape(self.x_out, marks[-(1)][1], interpolate=interpolate, area=area)

        self.loss += loss__
        self.x4 = x__
        self.area = area__

        self.loss1 = loss1_
        self.loss2 = loss2_
        self.loss3 = loss3_
        self.loss4 = loss4_

        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01) #tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
        self.opt_op = self.optimizer.minimize(self.loss)

    def train(self, sess, L, f):
        feed_dict = construct_feed_dict(f, L, placeholder=self.placeholder)
        shape, loss, _, l1, l2, l3, l4= sess.run([self.shape, self.loss, self.opt_op, self.loss1, self.loss2, self.loss3, self.loss4], feed_dict=feed_dict)
        return shape, loss, l1, l2, l3, l4

    def reconstruct(self, sess, L, feature):
        feed_dict = construct_feed_dict(feature, L, placeholder=self.placeholder)
        out, x1, x2, x3, x4, area, l1, l2, l3, l4 = sess.run([self.x_out, self.x1, self.x2, self.x3, self.x4, self.area, self.loss1, self.loss2, self.loss3, self.loss4], feed_dict=feed_dict)
        print(out.shape, 'out')
        print(x1.shape, 'x1')
        print(x2.shape, 'x2')
        print(x3.shape, 'x3')
        print(x4.shape, 'x4')
        print(area, 'area')
        print(l1, 'l1')
        print(l2, 'l2')
        print(l3, 'l3')
        print(l4, 'l4')
        print(out)
        return out   #out   x4

def GraphConvolution (input_layer, support, shape, act = tf.nn.sigmoid, sparse_inputs = False):
    vars = {}
    for i in range(len(support)):
        vars['weights_'+str(i)] = init_weight(shape=shape)
    bias = init_bias(shape=shape[1])
    vars['bias'] = bias
    supports = list()
    x = input_layer
    for i in range(len(support)):
        pre_sup = dot(x, vars['weights_' + str(i)], sparse=sparse_inputs)
        tem_support = dot(support[i], pre_sup, sparse=True)
        supports.append(tem_support)
    output = tf.add_n(supports)
    output += bias
    return act(output), vars

def init_weight(shape):
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    init_range = 0.5
    initial = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    weight = tf.Variable(initial)
    return weight

def init_bias(shape):
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial)

def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse.sparse_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

def construct_feed_dict(feature, support,placeholder):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholder['feature']: feature})
    for i in range(len(support)):
        feed_dict.update({placeholder['L'][i][j]: support[i][j] for j in range(len(support[i]))})
    return feed_dict

def min_distance(point,line):
    distance = tf.reduce_sum(tf.pow(tf.subtract(point, line),2.0),1)
    min_value = tf.gather(distance,tf.argmin(distance))
    return min_value

