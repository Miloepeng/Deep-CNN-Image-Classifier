# Importing all necessary libraries
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
# data = tf.keras.utils.image_dataset_from_directory('Data')
# data_iterator = data.as_numpy_iterator()
# batch = data_iterator.next()

#figure out what 1 and 0 are
# fig, ax = plt.subplots(ncols=4,figsize=(20,20))
# for idx, img in enumerate(batch[0][:4]):
#     ax[idx].imshow(img.astype(int))
#     ax[idx].title.set_text(batch[1][idx])
# plt.show()
#0 means Diabetic

#data pre-processing
# data = data.map(lambda x,y:(x/255,y))#optimize rgb values between 0-1
# length = len(data) #total of 20 batches of data
# train_size = int(length*0.7)
# val_size = int(length*0.2)
# test_size = int(length*0.1) #totals up to 20
# train = data.take(train_size)
# val = data.skip(train_size).take(val_size)
# test = data.skip(train_size+val_size).take(test_size)

#model building
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
# model =  Sequential()
# model.add(Conv2D(16,(3,3),1,activation = 'relu',input_shape=(256,256,3)))
# model.add(MaxPooling2D())

# model.add(Conv2D(32,(3,3),1,activation = 'relu'))
# model.add(MaxPooling2D())

# model.add(Conv2D(16,(3,3),1,activation = 'relu'))
# model.add(MaxPooling2D())

# model.add(Flatten())

# model.add(Dense(256,activation = 'relu'))
# model.add(Dense(1,activation = 'sigmoid'))

# model.compile('adam',loss=tf.losses.BinaryCrossentropy(),metrics=['accuracy'])

# #Training
# logdir = "Logs"
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir) 
# hist = model.fit(train,epochs = 20,validation_data = val,callbacks = [tensorboard_callback])

#plotting performance
# fig = plt.figure()
# plt.plot(hist.history['loss'],color = 'teal',label = 'loss')
# plt.plot(hist.history['val_loss'],color='orange',label = 'val_loss')
# fig.suptitle('Loss',fontsize=20)
# plt.legend(loc='upper left')
# plt.show()

# evaluating
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
# pre = Precision()
# re = Recall()
# acc = BinaryAccuracy()
# for batch in test.as_numpy_iterator():
#     x,y = batch
#     yhat = model.predict(x)
#     pre.update_state(y,yhat)
#     re.update_state(y,yhat)
#     acc.update_state(y,yhat)
import cv2
img = cv2.imread('5_0.jpg')
resize = tf.image.resize(img,(256,256))
# yhat = model.predict(np.expand_dims(resize/255,0))
# if yhat >0.5 :
#     print("Not Happy",yhat)
# else:
#     print("Happy",yhat)

#saving model
from tensorflow.keras.models import load_model
# model.save(os.path.join('models','diabeticmodel.h5'))
new_model = load_model(os.path.join('models','diabeticmodel.h5'))
print(new_model.predict(np.expand_dims(resize/255,0)))
# print(np.expand_dims(resize,0))
yhat = new_model.predict(np.expand_dims(resize/255,0))
if yhat >0.5 :
    print("Not Diabetic",yhat)
else:
    print("Diabetic",yhat)