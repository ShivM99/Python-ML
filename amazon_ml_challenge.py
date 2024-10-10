# -*- coding: utf-8 -*-
"""Amazon_ML_Challenge.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/15z0QEgPe6fUGWkEWQFqKvJFXcL6iykx2

**Importing the libraries**
"""

import numpy as np
import pandas as pd
import requests as req
import imageio as io
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # To be sure that truncated image do not raise an error
import cv2
from google.colab.patches import cv2_imshow
from multiprocessing import Pool

"""**Loading the dataset**"""

data = pd.read_csv ('train.csv')
print (data.head ())
print (data.shape)

"""**Trying preprocessing on a small set of images**"""

img = cv2.cvtColor (io.v2.imread (data.iloc [1, 0]), cv2.COLOR_BGR2RGB)
cv2_imshow (img)
img = cv2.cvtColor (img, cv2.COLOR_RGB2GRAY)
cv2_imshow (img)
img = cv2.bitwise_not (img)
cv2_imshow (img)
img = cv2.adaptiveThreshold (img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
kernel = np.ones ((1, 1))
img = cv2.medianBlur (cv2.morphologyEx (img, cv2.MORPH_CLOSE, kernel), 3)
cv2_imshow (img)

for row in range (0, 10):
    img = cv2.cvtColor (io.v2.imread (data.iloc [row, 0]), cv2.COLOR_BGR2RGB)
    cv2_imshow (img)
    img = cv2.cvtColor (img, cv2.COLOR_RGB2GRAY)
    cv2_imshow (img)
    img = cv2.bitwise_not (img)
    cv2_imshow (img)
    img = cv2.adaptiveThreshold (img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
    kernel = np.ones ((1, 1))
    img = cv2.medianBlur (cv2.morphologyEx (img, cv2.MORPH_CLOSE, kernel), 3)
    cv2_imshow (img)

"""**Extracting images and preprocessing them one-by-one**"""

row = 54 # Implementing preprocessing on 1 image
url = data.iloc [row, 0]
entity = data.iloc [row, 2]
value = data.iloc [row, 3]
filename = f'{row}_{entity}_{value}.jpg'
res = req.get (url, stream=True)
if res.status_code == 200:
    with open (filename, 'wb') as f:
        f.write (res.content)
        img = cv2.imread (filename)
        img = cv2.bitwise_not (cv2.cvtColor (img, cv2.COLOR_BGR2GRAY))
        cv2.imwrite (filename, img)
else:
    print(f'Failed to download {row} image')

def image_extraction (row):
    url = data.iloc [row, 0]
    entity = data.iloc [row, 2]
    value = data.iloc [row, 3]
    filename = f'{row}_{entity}_{value}.jpg'
    res = req.get (url, stream=True)
    if res.status_code == 200:
        with open ('filenames.txt', 'a') as names:
            names.write (f'{filename}\t{entity}\t{value}\n')
            with open (filename, 'wb') as f:
                f.write (res.content)
                img = cv2.imread (filename)
                img = cv2.bitwise_not (cv2.cvtColor (img, cv2.COLOR_BGR2GRAY))
                cv2.imwrite (filename, img)
                return
    else:
        print(f'Failed to download {row} image')
        return

if __name__ == '__main__':
    with Pool() as pool:
        for result in pool.imap (image_extraction, range (data.shape [0])): # Implementing preprocessing on all the 2.5 lakh images
            pass

"""**Final model on preprocessed data for OCR**"""

!pip install easyocr

import easyocr as ocr
reader = ocr.Reader (['en'], gpu=True)
import matplotlib.pyplot as plt
import re

img = cv2.imread ('/content/12_item_weight_3.53 ounce.jpg')
img = cv2.bitwise_not (cv2.cvtColor (img, cv2.COLOR_BGR2GRAY))
cv2_imshow (img)
result = reader.readtext (img)
for res in result:
    print (res)

entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram',
        'kilogram',
        'microgram',
        'milligram',
        'ounce',
        'pound',
        'ton'},
    'maximum_weight_recommendation': {'gram',
        'kilogram',
        'microgram',
        'milligram',
        'ounce',
        'pound',
        'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre',
        'cubic foot',
        'cubic inch',
        'cup',
        'decilitre',
        'fluid ounce',
        'gallon',
        'imperial gallon',
        'litre',
        'microlitre',
        'millilitre',
        'pint',
        'quart'}
}
allowed_units = {unit for entity in entity_unit_map for unit in entity_unit_map[entity]} # Creating a set of all units

patterns = {
    'item_weight': r'(\d+\.?\d*)\s*(g|gram|kg|kilogram|microgram|mg|milligram|oz|ounce|lb|pound|ton)',
    'maximum_weight_recommendation': r'(\d+\.?\d*)\s*(g|gram|kg|kilogram|microgram|mg|milligram|oz|ounce|lb|pound|ton)',
    'item_volume': r'(\d+\.?\d*)\s*(centilitre|cubic foot|cubic inch|cup|decilitre|fluid ounce|gallon|imperial gallon|litre|microlitre|millilitre|pint|quart)',
    'width': r'(\d+\.?\d*)\s*(cm|centimetre|foot|inch|m|metre|mm|millimetre|yard)',
    'depth': r'(\d+\.?\d*)\s*(cm|centimetre|foot|inch|m|metre|mm|millimetre|yard)',
    'height': r'(\d+\.?\d*)\s*(cm|centimetre|foot|inch|m|metre|mm|millimetre|yard)',
    'wattage': r'(\d+\.?\d*)\s*(kw|kilowatt|w|watt)',
    'voltage': r'(\d+\.?\d*)\s*(kv|kilovolt|mv|millivolt|v|volt)'
} # (\d+\.?\d*) means >=1 digits, followed by none or a single decimal, followed by >=0 digits

prev_id = data.iloc [0, 1]
data['predicted_value'] = ''
for row in range (2500): # Implementing preprocessing on the 1st 2500 images
    url = data.iloc [row, 0]
    group_id = data.iloc [row, 1]
    entity = data.iloc [row, 2]
    value = data.iloc [row, 3]
    filename = f'{row}_{entity}_{value}.jpg'
    res = req.get (url, stream=True)
    if res.status_code == 200:
        with open (filename, 'wb') as f:
            f.write (res.content)
            img = cv2.imread (filename)
            img = cv2.bitwise_not (cv2.cvtColor (img, cv2.COLOR_BGR2GRAY))
            if row > 0 and group_id == prev_id and ocr_value != None:
                print (row, '\t', ocr_value.group())
                data.iloc[row, 4] = ocr_value.group()
                continue
            prev_id = group_id
            result = reader.readtext (img)
            if entity in patterns.keys():
                for res in result:
                    text = res [1].lower()
                    if 'ter' in text: text = text.replace ('ter', 'tre')
                    elif 'feet' in text: text = text.replace ('feet', 'foot')
                    ocr_value = re.search (patterns[entity], text)
                    if ocr_value != None:
                        magnitude = re.search (r'(\d+\.?\d*)', ocr_value.group()).group()
                        if magnitude in filename:
                            print (row, '\t', ocr_value.group())
                            data.iloc[row, 4] = ocr_value.group()
                            break
                    elif res == result[-1] and ocr_value == None:
                        print (row, '\t')
                        data.iloc[row, 4] = ''
    else:
        print (f'Failed to download {row} image')
