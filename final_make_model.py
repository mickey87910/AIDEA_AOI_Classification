#!/usr/bin/python3.6
import pandas as pd
import numpy as np
import os
from sklearn.model_selection._split import train_test_split
from tensorflow._api.v1.keras.applications.vgg19 import VGG19
from tensorflow._api.v1.keras.applications.vgg19 import preprocess_input
# from tensorflow._api.v1.keras.applications.resnet50 import ResNet50
# from tensorflow._api.v1.keras.applications.resnet50 import preprocess_input
from tensorflow._api.v1.keras.preprocessing import image
from tensorflow._api.v1.keras.layers import Input,Dense,Dropout
from tensorflow._api.v1.keras.optimizers import Adam
from tensorflow._api.v1.keras.models import Model,Sequential
img_csv = pd.read_csv("train.csv")
id_array = []
label_array = []
##分類
for i in img_csv['ID']:
    id_array.append(i)
for i in img_csv['Label']:
    label_array.append(i)
##進行圖片之處理
img_array=[]
img_directory = "./train_images_resize/"
for i in id_array:
    image_path = img_directory+i
    print("Loading images : "+image_path)
    the_image = image.load_img(image_path,target_size=(224,224,3))
    x = image.img_to_array(the_image)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    img_array.append(x)
img_data = np.array(img_array)
print(img_data.shape)
img_data = np.rollaxis(img_data,1,0)
print(img_data.shape)
img_data=img_data[0]
print(img_data.shape)
num_classes = 6
train_X,test_X,train_Y,test_Y = train_test_split(img_data,label_array,test_size=0.10)
image_input = Input(shape=(224,224,3))
model = VGG19(input_tensor=image_input,include_top=True,weights=None)
last_layer = model.get_layer('fc2').output  #VGG
# last_layer = model.get_layer('fc1000').output  #RES
# last_layer = model.get_layer('predictions').output
last_layer = Dropout(0.5)(last_layer)
out = Dense(num_classes,activation="softmax", name="output")(last_layer)
model  = Model(image_input,out)
for layer in model.layers[:-1]:
     layer.trainable = False
model.summary()

model.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),metrics=['accuracy'])


hist = model.fit(train_X,train_Y,
        batch_size=16,
        epochs=30,
        verbose=1,
        validation_data=(test_X,test_Y)
)
(loss,accuracy)=model.evaluate(test_X,test_Y,batch_size=8,verbose=1)
prediction = model.predict(test_X)
predict = np.argmax(prediction,axis=1)
print("Save Model?y/n")
a=input()
if(a=="y"):
        print("Saving Model")
        model.save("RESRetry14.h5",overwrite=True,include_optimizer=True)
else:
        print("Over")
