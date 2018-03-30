
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Dense
from keras.layers import Dropout,Activation,Flatten
import numpy as np
import matplotlib.pyplot as plt 

%matplotlib inline
############################################

NUM_CLASSES =10
(X_train ,y_train),(X_test,y_test)=cifar10.load_data()
#####################################################

X_train.shape
##############################################

X_train = X_train.astype('float32')/255.0
X_test =X_test.astype('float32')/255.0

y_train=keras.utils.to_categoricaly(y_train,NUM_CLASSES)
y_test =keras.utils.to_categoricaly(y_test,NUM_CLASSES)
#############################################

TYPE_MAP={
	0:u'Uçak',
	1:'Otomobil',
	2:u'Kuş',
	3:'Kedi',
	4:u'Köpek',
	5:u'Köpek',
	6:u'Kurbağa',
	7:'At',
	8:'Gemi',
	9:'Kamyon',
}

###############################################
dataset_size=X_train.shape[0]

idx=random.randint(0,dataset_size)

sample_img=X_train[idx]
sample_label=np.argmax(y_train[idx])

print(u"Resmin cinsi:%S" % TYPE_MAP[sample_label])
plt.imshow(sample_img)
###############################################

input_shape =X_train.shape[1:]

model=Sequential()

model.add(Conv2D(32,(3,3),padding='same',input_shape=input_shape,activaion='relu'))
model.add(Activation('relu'))

model.add(Conv2D(32,(3,3)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(64,(3,3)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(1024,activaion='relu'))
model.add(Dropout(0.25))

model.add(Dense(128,activation='relu'))
model.add(Dense(NUM_CLASSES,activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizer.rmsprop(lr=1e-4,decay=1e-6),metrics=['accuracy'])
###################################################################

mode.fit(X_train,y_train,batch_size=10,epochs=10,validation_data=(X_test,y_test))
#################################################################

model.evaluate(X_test,y_test,verbose=1)
#####################################################################

idx=random.randint(0,X_test.shape[0])
sample_img=X_test[idx]
sample_label=np.argmax(y_train[idx])

print(u"Resmin cinsi: %S" % TYPE_MAP[sample_label])
plt.imshow(sample_img)

sample_img=sample_img.reshape(1,32,32,3)
for i,v in enumerate(model.predict(sample_img)[0]):
	print(u'resmin %s olma ihtimali: %.6f%%' % (TYPE_MAP[i],v *100))

##################################


