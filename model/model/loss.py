# create custom loss function for training
import sys
sys.path.append('./model/utils/')
import keras.backend as K
from keras.layers import Lambda
import tensorflow as tf
import utils

def audio_discriminate_loss(gamma=0.1,people_num=2):
    def loss_func(S_true,S_pred,gamma=gamma,people_num=people_num):
        sum = 0
        for i in range(people_num):
            sum += K.sum(K.flatten((K.square(S_true[:,:,:,i]-S_pred[:,:,:,i]))))
            for j in range(people_num):
                if i != j:
                    sum -= gamma*K.sum(K.flatten((K.square(S_true[:,:,:,i]-S_pred[:,:,:,j]))))

        loss = sum / (people_num*298*257*2)
        return loss
    return loss_func


def audio_discriminate_loss2(gamma=0.1,beta = 2*0.1,people_num=2):
    def loss_func(S_true,S_pred,gamma=gamma,beta=beta,people_num=people_num):
        sum_mtr = K.zeros_like(S_true[:,:,:,:,0])
        for i in range(people_num):
            sum_mtr += K.square(S_true[:,:,:,:,i]-S_pred[:,:,:,:,i])
            for j in range(people_num):
                if i != j:
                    sum_mtr -= gamma*(K.square(S_true[:,:,:,:,i]-S_pred[:,:,:,:,j]))

        for i in range(people_num):
            for j in range(i+1,people_num):
                #sum_mtr -= beta*K.square(S_pred[:,:,:,i]-S_pred[:,:,:,j])
                #sum_mtr += beta*K.square(S_true[:,:,:,:,i]-S_true[:,:,:,:,j])
                pass
        #sum = K.sum(K.maximum(K.flatten(sum_mtr),0))

        loss = K.mean(K.flatten(sum_mtr))

        return loss
    return loss_func


def cRM_tanh_recover(O, K=10, C=0.1):
    # numerator = K - O
    # denominator = K + O
    K = tf.constant(K, dtype=tf.float32)
    numerator = tf.math.add(-O,K)
    denominator = tf.math.add(O,K)

    # M = -K.multiply((1.0 / C), K.log(K.divide(numerator, denominator)))
    # a = Lambda(lambda inputs: inputs[0] / inputs[1])([numerator, denominator])
    ratio = tf.constant(1.0 / C, dtype=tf.float32)
    M = tf.multiply(ratio, tf.math.log(tf.math.divide(numerator, denominator)))
    return M


def fast_icRM(Y, crm, K=10, C=0.1):
    M = cRM_tanh_recover(crm, K, C)
    S = tf.zeros((M.shape[0], M.shape[1], 1))
    # S[:, :, 0] = K.multiply(M[:, :, 0], Y[:, :, 0]) - K.multiply(M[:, :, 1], Y[:, :, 1])
    # S[:, :, 1] = K.multiply(M[:, :, 0], Y[:, :, 1]) + K.multiply(M[:, :, 1], Y[:, :, 0])

    # S1 = Lambda(lambda x: x[0] * x[1])([M[:, :, 0], Y[:, :, 0]]) - Lambda(lambda x: x[0] * x[1])([M[:, :, 1], Y[:, :, 1]])
    S1 = tf.multiply(M[:, :, 0], Y[:, :, 0]) - tf.multiply(M[:, :, 1], Y[:, :, 1])
    # S2 = Lambda(lambda x: x[0] * x[1])([M[:, :, 0], Y[:, :, 1]]) + Lambda(lambda x: x[0] * x[1])([M[:, :, 1], Y[:, :, 0]])
    S2 = tf.multiply(M[:, :, 0], Y[:, :, 1]) - tf.multiply(M[:, :, 1], Y[:, :, 0])
    return tf.concat([tf.expand_dims(S1, 2), tf.expand_dims(S2, 2)], axis=2)



def audio_discriminate_original(people_num=2, batch_size=2):
    def loss_func(S_true, S_pred, people_num=people_num, batch_size=batch_size):
        sum_mtr = tf.zeros_like(S_true[0,:,:,:,0])
        for i in range(batch_size):
            for j in range(people_num):
                pred = fast_icRM(S_true[i,:,:,:,-1], S_pred[i,:,:,:,j])
                sum_mtr += tf.math.square(S_true[i,:,:,:,j]-pred)
            # for j in range(people_num):
            #     if i != j:
            #         sum_mtr -= gamma*(K.square(S_true[:,:,:,:,i]-S_pred[:,:,:,:,j]))

        for i in range(people_num):
            for j in range(i+1,people_num):
                #sum_mtr -= beta*K.square(S_pred[:,:,:,i]-S_pred[:,:,:,j])
                #sum_mtr += beta*K.square(S_true[:,:,:,:,i]-S_true[:,:,:,:,j])
                pass
        #sum = K.sum(K.maximum(K.flatten(sum_mtr),0))
        loss = tf.math.reduce_mean(tf.reshape(sum_mtr, [-1]))

        return loss
    return loss_func





