#!/usr/bin/python3.6
import pandas as pd
import numpy as np
from tensorflow._api.v1.keras.preprocessing import image
from tensorflow._api.v1.keras.applications.vgg16 import preprocess_input
from tensorflow._api.v1.keras.models import load_model
csv_data = pd.read_csv("test.csv")
id_array =[]
for i in csv_data['ID']:
    id_array.append(i)
img_array=[]
img_directory = "./test_images_resize/"
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
model =load_model("GORES.h5")
print("預測開始")
prediction = model.predict(img_data)
predict = np.argmax(prediction,axis=1)
print("預測結束")
for i in range(len(img_data)):
    csv_data.loc[i,'Label'] = str(predict[i])[0]
print(csv_data)
csv_data.to_csv("GORES.csv",header=True,index=False)
