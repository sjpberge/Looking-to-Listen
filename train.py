import sys

sys.path.append('./model/model/')
import AV_model as AV
from option import ModelMGPU, latest_file
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from keras.models import Model, load_model
from data_load import AVGenerator
from keras.callbacks import TensorBoard
from keras import optimizers
import os
from loss import audio_discriminate_original as audio_loss_original
from loss import audio_discriminate_loss2 as audio_loss
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K

# Resume Model
resume_state = False
tf.compat.v1.enable_eager_execution()
tf.config.experimental_run_functions_eagerly(True)

# Parameters
people_num = 2
epochs = 1
initial_epoch = 0
batch_size = 1
gamma_loss = 0.1
beta_loss = gamma_loss * 2

# Accelerate Training Process
workers = 8
MultiProcess = True
NUM_GPU = 0

# PATH
model_path = './saved_AV_models'  # model path
chkpt_path = './tf_ckpts'
database_path = 'data/'

# create folder to save models
folder = os.path.exists(model_path)
if not folder:
    os.makedirs(model_path)
    print('create folder to save models')

folder = os.path.exists(chkpt_path)
if not folder:
    os.makedirs(chkpt_path)
    print('create folder to save checkpoints')
filepath = model_path + "/AVmodel-" + str(people_num) + "p-{epoch:03d}-{val_loss:.5f}.h5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# checkpoint = ModelCheckpoint(filepath, save_freq='epoch')
# checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq='epoch')

# format: mix.npy single.npy single.npy
trainfile = []
valfile = []

with open((database_path + 'AVdataset_train.txt'), 'r') as t:
    trainfile = t.readlines()
with open((database_path + 'AVdataset_val.txt'), 'r') as v:
    valfile = v.readlines()

# the training steps
if resume_state:
    if NUM_GPU > 1:
        latest_file = latest_file(model_path + '/')
        AV_model = load_model(latest_file, custom_objects={"tf": tf, 'loss': audio_loss_original(batch_size=batch_size, people_num=people_num)})
        # K.set_value(AV_model.optimizer.learning_rate, 0.00003)
    else:
        latest_file = latest_file(model_path + '/')
        AV_model = load_model(latest_file, custom_objects={"tf": tf, 'loss': audio_loss_original(batch_size=batch_size, people_num=people_num)})
        # K.set_value(AV_model.optimizer.learning_rate, 0.00003)
else:
    if NUM_GPU > 1:
        with tf.device('/cpu:0'):
            AV_model = AV.AV_model(people_num)
        parallel_model = ModelMGPU(AV_model, NUM_GPU)
        adam = optimizers.Adam(learning_rate=0.00003)
        loss = audio_loss_original(batch_size=batch_size, people_num=people_num)
        parallel_model.compile(loss=loss, optimizer=adam)
    else:
        AV_model = AV.AV_model(people_num)
        adam = optimizers.Adam(learning_rate=0.00003)
        loss = audio_loss_original(batch_size=batch_size, people_num=people_num)
        AV_model.compile(optimizer=adam, loss=loss)

train_generator = AVGenerator(trainfile, database_path=database_path, batch_size=batch_size, shuffle=True)
val_generator = AVGenerator(valfile, database_path=database_path, batch_size=batch_size, shuffle=True)

if NUM_GPU > 1:
    # learning rate reduced by half every 1.8 million batches, total of 5 million batches of 6, 30 million examples, 1.5 million batches
    # lr reduced by half twice in the course of the training
    parallel_model = ModelMGPU(AV_model, NUM_GPU)
    adam = optimizers.Adam(learning_rate=0.00003)
    loss = audio_loss(gamma=gamma_loss, beta=beta_loss, people_num=people_num)
    parallel_model.compile(loss=loss, optimizer=adam)
    print(AV_model.summary())
    history = parallel_model.fit_generator(generator=train_generator,
                                 validation_data=val_generator,
                                 epochs=epochs,
                                 workers=workers,
                                 use_multiprocessing=MultiProcess,
                                 callbacks=[TensorBoard(log_dir='./log_AV', update_freq=10000, histogram_freq=1)]
                                # callbacks = [TensorBoard(log_dir='./log_AV'), checkpoint, rlr]

    )

    filepath = model_path + "/AVmodel-" + str(people_num) + ".h5"
    AV_model.save_weights('./checkpoints/my_checkpoint')

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss plt.plot(history.history['loss']) plt.plot(history.history['val_loss']) plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

if NUM_GPU <= 1:
    history=AV_model.fit_generator(generator=train_generator,
                           validation_data=val_generator,
                           epochs=epochs,
                           workers=workers,
                           use_multiprocessing=MultiProcess,
                           # callbacks=[TensorBoard(log_dir='./log_AV'), checkpoint, rlr],
                           callbacks=[TensorBoard(log_dir='./log_AV', update_freq=10000, histogram_freq=1)]
                           )

    filepath = model_path + "/AVmodel-" + str(people_num)
    AV_model.save(filepath)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss plt.plot(history.history['loss']) plt.plot(history.history['val_loss']) plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
