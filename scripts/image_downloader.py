"""
    This is a script that takes data from the CSVs, download and stores the images.
    It assumes the csv has the following columns: name, thumbnail link, date, img link, subcategory id, subcategory

    The final dataset is stored in the final_filename{i} files according to CIFAR semantics:

    The archive contains the files dataset1, dataset2, ... Each of these files is a Python "pickled" object produced with Pickle. 
    Here is a python3 routine which will open such a file and return a dictionary:

    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    Loaded in this way, each of the batch files contains a dictionary with the following elements:
    data -- a 10000x7500 numpy array of uint8s. Each row of the array stores a 50x50 colour image. The first 2500 entries contain the red channel values, the next 2500 the green, and the final 2500 the blue.
    The image is stored in row-major order, so that the first 50 entries of the array are the red channel values of the first row of the image.
    labels -- a list of 10000 numbers in the range 0-20. The number at index i indicates the label of the ith image in the array data.

    Dependencies: Pillow, Pickle, numpy, requests
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
from time import sleep
class Tags(Enum):
    NAME = 0
    THUMBNAIL = 1
    DATE = 2
    IMG = 3
    SUB_ID = 4
    SUB = 5

def download_file_images(file, data, labels, lock: threading.RLock):
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
                    try:
                        answer = requests.get(thumb_url, headers=headers) #download thumbnail
                        if answer.status_code <= 400:
                            image = answer.content
                            #resize image
                            image = Image.open(io.BytesIO(image))
                            # image.save(thumb_url.split("/")[-1]) #was used to make sure the images are stored correctly
                            image_size = (max(image.size), max(image.size))
                            image = ImageOps.pad(image, image_size, color="white")
                            image.save(thumb_url.split("/")[-1]) #was used to make sure the images are stored correctly
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
                        else:
                            print(f"Received a non-success code: {answer.status_code}")
                            sleep(10)
                    except Exception as e:
                        print(f'Caught the following exception: {e}')

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
    worker = threading.Thread(target=download_file_images, args=(file, data, labels, threading.RLock(), ))
    worker.start()
    workers.append(worker)

for thread in workers:
    thread.join()
    