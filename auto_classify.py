#################################################################################################################################################################
import cv2
import torch
import os
import pandas as pd
#################################################################################################################################################################
def Yolo_darknet(results,imges_list):
    df_list = [pd.DataFrame() for _ in range(len(imges_list))]
    for i in range(len(imges_list)):
        df_list[i]['class'] = results.pandas().xyxy[i]['class']
        df_list[i]['x_centre'] = (results.pandas().xyxy[i]['xmin'] + results.pandas().xyxy[i]['xmax']) / float(2 * imges_list[i].shape[1])
        df_list[i]['y_centre'] = (results.pandas().xyxy[i]['ymin'] + results.pandas().xyxy[i]['ymax']) / float(2 * imges_list[i].shape[0])
        df_list[i]['X_width'] = abs((results.pandas().xyxy[i]['xmin'] - results.pandas().xyxy[i]['xmax'])) / imges_list[i].shape[1]
        df_list[i]['y_height'] = abs((results.pandas().xyxy[i]['ymin'] - results.pandas().xyxy[i]['ymax'])) / imges_list[i].shape[0]
    return df_list
#################################################################################################################################################################
def output_file(df_list,batch_images_names,ouput_path):
    for i,img in enumerate(batch_images_names):
        if not df_list[i].empty:
            with open(ouput_path + img.split(".")[0] + ".txt", 'a') as f:
                df_string =df_list[i].to_string(header=False, index=False)
                f.write(df_string)
                f.write("\n")
#################################################################################################################################################################
def load_list_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()
        content = [line.strip() for line in content]
        return content
#################################################################################################################################################################
def load_images(images_path):
    img_names = os.listdir(images_path)
    imges_list = []
    for img in img_names:
        imges_list.append(cv2.imread(images_path + img)[..., ::-1])
    return imges_list,img_names
#################################################################################################################################################################
def run_your_code(im_size,conf_thres):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_directory, 'output/YOLO_darknet/')
    images_path = os.path.join(current_directory, 'input/')
    imges_list,img_names = load_images(images_path)
    model_path = os.path.join(current_directory, 'model/model.pt')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    batch_size = 10
    num_images = len(imges_list)
    start_idx = 0
    model.conf = conf_thres
    while start_idx < num_images:
        end_idx = min(start_idx + batch_size, num_images)
        batch_images = imges_list[start_idx:end_idx]
        batch_images_names = img_names[start_idx:end_idx]
        results = model(batch_images,size=im_size)
        df_list = Yolo_darknet(results, batch_images)
        output_file(df_list, batch_images_names, output_path)
        start_idx += batch_size
#################################################################################################################################################################