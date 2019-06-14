#!/usr/bin/python3.6
import cv2
import pandas as pd
csv_data = pd.read_csv("train.csv")
id_array =[]
for i in csv_data['ID']:
    id_array.append(i)
img_directory = "./train_images/"
for i in id_array:
    image_path = img_directory+i
    print("Loading images : "+image_path)
    img = cv2.imread(image_path)
    x = cv2.resize(img,(224,224),interpolation=cv2.INTER_AREA)
    cv2.imwrite("./train_images_resize/"+i,x)
    