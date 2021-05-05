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

class Tags(Enum):
    NAME = 0
    THUMBNAIL = 1
    DATE = 2
    IMG = 3
    SUB_ID = 4
    SUB = 5

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

data = np.empty(0, dtype=object) #todo: remove dtype=object once resize has been implemented
labels = np.empty(0)
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:59.0) Gecko/20100101 Firefox/59.0'} #needed, otherwise the request hangs
#todo: is there a workaround??

#files are in the dataset folder
for file in csv_files:
    print("Parsing " + file)
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
                    print("received answer")
                    if answer.status_code != 404:
                        image = answer.content
                        #todo: resize image

                        #store resized image
                        data = np.append(data, image) #todo: check they append in the right axis
                        labels = np.append(labels, GET_CATEGORY(file))
                        collected_samples += 1

                        #check if collected enough samples
                        if collected_samples % samples4file == 0:
                            with open(FINAL_DATASET_PATH(), "w") as data_file:
                                dict2write = {"data": data, "labels": labels}
                                pickle.dump(dict2write, data_file, protocol=pickle.HIGHEST_PROTOCOL)
