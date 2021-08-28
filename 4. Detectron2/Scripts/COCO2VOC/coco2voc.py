#encoding:utf-8

import os
import json
import cv2
from lxml import etree
import xml.etree.cElementTree as ET
import time
import pandas as pd
from tqdm import tqdm
import json
import argparse




def coco2voc(anno,xml_dir):

    with open(anno, 'r', encoding='utf-8') as load_f:
        f = json.load(load_f)


    imgs = f['images']

    df_cate = pd.DataFrame(f['categories'])
    _ = df_cate.sort_values(["id"],ascending=True)
    df_anno = pd.DataFrame(f['annotations'])
    categories = dict(zip(df_cate.id.values, df_cate.name.values))

    for i in tqdm(range(len(imgs))):
        xml_content = []
        file_name = imgs[i]['file_name']
        height = imgs[i]['height']
        img_id = imgs[i]['id']
        width = imgs[i]['width']

        xml_content.append("<annotation>")
        xml_content.append("	<folder>VOC2007</folder>")
        xml_content.append("	<filename>"+file_name+"</filename>")
        xml_content.append("	<size>")
        xml_content.append("		<width>"+str(width)+"</width>")
        xml_content.append("		<height>"+str(height)+"</height>")
        xml_content.append("	</size>")
        xml_content.append("	<segmented>0</segmented>")
        
        annos = df_anno[df_anno["image_id"].isin([img_id])]

        for index, row in annos.iterrows():
            bbox = row["bbox"]
            category_id = row["category_id"]
            cate_name = categories[category_id]

            
            xml_content.append("	<object>")
            xml_content.append("		<name>"+cate_name+"</name>")
            xml_content.append("		<pose>Unspecified</pose>")
            xml_content.append("		<truncated>0</truncated>")
            xml_content.append("		<difficult>0</difficult>")
            xml_content.append("		<bndbox>")
            xml_content.append("			<xmin>"+str(int(bbox[0]))+"</xmin>")
            xml_content.append("			<ymin>"+str(int(bbox[1]))+"</ymin>")
            xml_content.append("			<xmax>"+str(int(bbox[0]+bbox[2]))+"</xmax>")
            xml_content.append("			<ymax>"+str(int(bbox[1]+bbox[3]))+"</ymax>")
            xml_content.append("		</bndbox>")
            xml_content.append("	</object>")
        xml_content.append("</annotation>")

        x = xml_content
        xml_content=[x[i] for i in range(0,len(x)) if x[i]!="\n"]


        xml_path = os.path.join(xml_dir,file_name.split('.')[-2] + '.xml')

        with open(xml_path, 'w+',encoding="utf8") as f:
            f.write('\n'.join(xml_content))
        xml_content[:]=[]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="convert coco .json annotation to voc .xml annotation")
    parser.add_argument('--json_path',type = str ,help = 'path to json file.',default="./annotaions/train.json")
    parser.add_argument('--output',type = str ,help = 'path to output xml files.',default="./train_xml")
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    coco2voc(args.json_path, args.output)