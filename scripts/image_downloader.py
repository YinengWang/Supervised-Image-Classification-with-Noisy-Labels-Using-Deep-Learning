"""
    This is a script that takes data from various CSV files, download and stores the images.
    It assumes the csv has the following columns: name, thumbnail link, date, img link, subcategory id, subcategory.
    The saved dataset will now have an equal distribution among the subcategories

    The final dataset can be store according to two different conventions:
        one that is more pytorch friendly: saves the images as individual files inside a folder and the labels in a csv
        the other is more numpy friendly: the images are stored as numpy array inside a file "pickled" from a dictionary with also the labels 
    
    Details of the numpy version:
        The final dataset is stored in the final_filename{i} files according to CIFAR semantics:
        The archive contains the files dataset1, dataset2, ... Each of these files is a Python "pickled" object produced with Pickle. 
        Here is a python3 routine which will open such a file and return a dictionary:

        def unpickle(file):
            import pickle
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict
        
        Loaded in this way, each of the batch files contains a dictionary with the following elements:
        data -- a 1000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image.
                The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.
                The image is stored in row-major order, so that the first 50 entries of the array are the red channel values of the first row of the image.
        labels -- a list of 1000 numbers in the range 1-21. The number at index i indicates the label of the ith image in the array data.
        sublabels -- a list of 1000 numbers in the range 1-?. The number at index i indecates the sublabel (detailed category) of the ith image in the array data.
                     The sublabel is an ID. It therefore needs enumeration before use.

    Details of the pytorch version:
        The images will all be stored inside the folder 'images' (that will be created if non-existent) with their original name (assumed to be unique)
        The labels will be stored inside a csv file dataset_labels.csv here the first column has the image name, the second its label, the third its sublabel

    Dependencies: Pillow, Pickle, numpy, requests
"""

from os import listdir
import csv
from enum import Enum
from typing import Any, Callable
import requests
import numpy as np
import pickle
import io
import PIL.Image as Image
import PIL.ImageOps as ImageOps
from threading import RLock
from concurrent.futures import ThreadPoolExecutor
from time import sleep
import argparse
import csv
from os import mkdir
from os.path import exists
from dataclasses import dataclass

class Tags(Enum):
    NAME = 0
    THUMBNAIL = 1
    DATE = 2
    IMG = 3
    SUB_ID = 4
    SUB = 5

class STORE_FORMAT(Enum):
    NUMPY_FRIENDLY = 0
    PYTORCH_FRIENDLY = 1

@dataclass
class DownloaderConfig():
    store_format: STORE_FORMAT = STORE_FORMAT.PYTORCH_FRIENDLY # the prefered store format
    dataset_folder: str = "./dataset/" # the folder in which to find the csv files and in which to store the dataset
    date_accepted: int = 2010 # date from which to save a product
    final_filename: str = "dataset" # the base name for the dataset that will be stored
    percentage2download: int = 100 # percentage of dataset to download (not garanteed)
    samples4category: int = 1000 # the number of samples to store for each category (if a category has less samples, it will not be stored)
    labels_writer: Any = None # used in the PYTORCH_FRIENDLY, its a csv writer for the labels file
    final_dataset_path: Callable = None # used in NUMPY_FRIENDLY, used to generate the name of the next dataset file

GET_CATEGORY = lambda x: x.split("/")[-1].split("_")[0]  # get the category from the filename
categories_written = 0 # the number of categories written

def store_data(config, data):
    """Store the given data, based on the config object

    Args:
        config (DownloaderConfig): an object with all the useful configs
        data (list): a list of tuple ((image, image_name), label, sublabel)
    """
    global categories_written
    if config.store_format == STORE_FORMAT.NUMPY_FRIENDLY:
        # store as numpy array
        labels = []
        sublabels = []
        images2write = []
        for ((image, image_name), label, sublabel) in data: 
            images2write.append(np.array(ImageOps.pad(Image.open(io.BytesIO(image)).convert('RGB'), (32, 32))))
            labels.append(label)
            sublabels.append(sublabel)

        with open(config.final_dataset_path(), "wb") as data_file:
            dict2write = {"data": images2write, "labels": np.array(labels), "sublabels": np.array(sublabels)}
            pickle.dump(dict2write, data_file, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        for ((image, image_name), label, sublabel) in data: 
            with open(config.dataset_folder + "images/" + image_name, "wb") as file:
                file.write(image)
            config.labels_writer.writerow([image_name, label, sublabel])
    categories_written += 1
        
        

def download_file_images(file, config, categories, lock, subcat):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:59.0) Gecko/20100101 Firefox/59.0'
    } # needed, otherwise the request hangs
    # todo: is there a workaround??

    with open(config.dataset_folder + file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # read each line
        for index, row in enumerate(reader):
            # take one in ten
            if index % (100 / config.percentage2download) == 0:
                date = row[Tags.DATE.value]
                # check date
                if not date or int(date.split("-")[0]) > config.date_accepted:
                    thumb_url = row[Tags.THUMBNAIL.value]
                    sublabel = row[Tags.SUB_ID.value]
                    download_image = True
                    with lock:
                        if not sublabel in subcat:
                            subcat[sublabel] = []
                        download_image = len(subcat[sublabel]) < config.samples4category
                    if download_image:
                        try:
                            answer = requests.get(thumb_url, headers=headers)  # download thumbnail
                            if answer.status_code <= 400:
                                image = answer.content
                                data = (image, thumb_url.split("/")[-1])
                                label = categories[GET_CATEGORY(file)]
                                
                                with lock:
                                    if len(subcat[sublabel]) < config.samples4category: 
                                        subcat[sublabel].append((data, label, sublabel))
                                        if len(subcat[sublabel]) == config.samples4category:
                                            store_data(config, subcat[sublabel]) # todo: could be improved as operation can be done without lock
                                            print(f'Storing {sublabel} subcat samples')
                            else:
                                print(f"Received a non-success code {answer.status_code} when crawling:")
                                print(thumb_url)
                                sleep(10)
                        except Exception as e: # todo: bad as it catches all the exceptions
                            print('Caught the following exception when crawling: ', e)
                            print(thumb_url)


def multithread_image_download(config, max_threads):
    all_files = listdir(config.dataset_folder)
    all_files.sort()
    csv_files = []
    categories = {}
    subcat = {}

    for file in all_files:
        if not file.endswith(".csv"):
            continue
        csv_files.append(file)
        if not GET_CATEGORY(file) in categories:
            categories[GET_CATEGORY(file)] = len(categories)

    common_lock = RLock()
    
    labels_csv_file = None
    if config.store_format == STORE_FORMAT.PYTORCH_FRIENDLY:
        if not exists(config.dataset_folder + "images"):
            mkdir(config.dataset_folder + "images")
        labels_csv_file = open(f'{config.dataset_folder}{config.final_filename}_lables.csv', "w", newline='')
        config.labels_writer = csv.writer(labels_csv_file, delimiter=',')

    # files are in the dataset folder
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        for file in reversed(csv_files):
            executor.submit(download_file_images, file, config, categories, common_lock, subcat)

    # store_data(config, data, labels, sublabels)
    if labels_csv_file:
        labels_csv_file.close()

if __name__ == "__main__":
    # key parameters
    downloader_config = DownloaderConfig()

    # accept key parameter as args
    parser = argparse.ArgumentParser(description='Downloads the images from the given csv files and stores them in the given format')
    parser.add_argument('--format', help='the format in which to save the images, either "numpy" or "pytorch". Default is "numpy"')
    parser.add_argument('--folder', help='the folder in which to find the csv files, default is "./dataset/"')
    parser.add_argument('--threads', help="the max number of threads running at the same time, default: uncapped")
    parser.add_argument('--dataset-percentage', help="the percentage of the dataset to download, default is 10")
    args = vars(parser.parse_args())
    
    # set arguments based on the parsed ones
    if args['format'] == "pytorch":
        downloader_config.store_format = STORE_FORMAT.PYTORCH_FRIENDLY
    elif args['format'] and args['format'] != "numpy":
        print('The format you provided is not valid.')
        parser.print_help()
        exit()
    
    if args['folder']:
        downloader_config.dataset_folder = args['folder']
        if downloader_config.dataset_folder[-1] != "/":
            downloader_config.dataset_folder += "/"
    max_threads = None
    if args['threads']:
        max_threads = int(args['threads'])

    if args['dataset_percentage']:
        downloader_config.percentage2download = args['dataset-percentage']
    
    # define lambda used in NUMPY_FRIDENLY to name the datasets name
    downloader_config.final_dataset_path = lambda: f'{downloader_config.dataset_folder}{downloader_config.final_filename}' +\
                                                   f'{categories_written}'

    multithread_image_download(downloader_config, max_threads)