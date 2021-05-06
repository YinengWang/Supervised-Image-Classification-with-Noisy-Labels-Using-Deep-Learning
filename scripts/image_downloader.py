"""
    This is a script that takes data from the CSVs, download and stores the images.
    It assumes the csv has the following columns: name, thumbnail link, date, img link, subcategory id, subcategory
"""

from os import listdir
import csv
from enum import Enum
import requests
import numpy as np
import pickle
import io
import PIL.Image as Image
import PIL.ImageOps as ImageOps
import threading

class Tags(Enum):
    NAME = 0
    THUMBNAIL = 1
    DATE = 2
    IMG = 3
    SUB_ID = 4
    SUB = 5

def download_file_images(file, data, labels, lock: threading.RLock, image_size):
    global collected_samples
    csv_filename = DATASET_DIR + file
    with open(csv_filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        #read each line
        for index, row in enumerate(reader):
            #take one in ten
            if index % (100 / percentage2download) == 0:
                date = row[Tags.DATE.value]
                #check date
                if not date or int(date.split("-")[0]) > date_accepted:
                    thumb_url = row[Tags.THUMBNAIL.value]
                    
                    answer = requests.get(thumb_url, headers=headers) #download thumbnail
                    if answer.status_code != 404:
                        image = answer.content
                        #resize image
                        image = Image.open(io.BytesIO(image))
                        image = ImageOps.pad(image, image_size)
                        image = np.array(image)
                        #store resized image
                        with lock:
                            data.append(image)
                            labels.append(categories[GET_CATEGORY(file)])
                            collected_samples += 1

                            #check if collected enough samples
                            if collected_samples % samples4file == 0:
                                with open(FINAL_DATASET_PATH(), "w") as data_file:
                                    dict2write = {"data": np.array(data), "labels": np.array(labels)}
                                    pickle.dump(dict2write, data_file, protocol=pickle.HIGHEST_PROTOCOL)
                                data.clear()
                                labels.clear()
                            if collected_samples % 100 == 0:
                                print(f'Collected {collected_samples} samples')


GET_CATEGORY = lambda x: x.split("_")[0] #get the category from the filename
DATASET_DIR = "./dataset/"


all_files = listdir(DATASET_DIR)
all_files.sort()
csv_files = []
categories = {}

#IMPORTANT PARAMETERS
percentage2download = 10 #10%
samples4file = 10000 #how many samples to store per file
final_filename = "dataset"
date_accepted = 2010 #date from which to collect samples
IMAGE_SIZE = (50, 50) #the size of the final image

collected_samples = 0
FINAL_DATASET_PATH = lambda: DATASET_DIR + final_filename + int(collected_samples / samples4file) #get dataset path

for file in all_files:
    if not file.endswith(".csv"):
        continue
    csv_files.append(file)
    if not GET_CATEGORY(file) in categories:
        categories[GET_CATEGORY(file)] = len(categories)

data = []
labels = []
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:59.0) Gecko/20100101 Firefox/59.0'} #needed, otherwise the request hangs
#todo: is there a workaround??

workers = []

#files are in the dataset folder
for file in reversed(csv_files):
    worker = threading.Thread(target=download_file_images, args=(file, data, labels, threading.RLock(), IMAGE_SIZE, ))
    worker.start()
    workers.append(worker)

for thread in workers:
    thread.join()
    