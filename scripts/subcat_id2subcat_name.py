"""
    A simple script that generates a csv file with the link between the subcategories IDs and their names.
    This information is gain from the various CSV files. 
    It is assumed that the CSVs have the following columns: name, thumbnail link, date, img link, subcategory id, subcategory.
"""

import csv
from os import listdir
from enum import Enum
import argparse

class Tags(Enum):
    NAME = 0
    THUMBNAIL = 1
    DATE = 2
    IMG = 3
    SUB_ID = 4
    SUB = 5


def generate_subcat2names(dataset_folder = '.'):
    all_files = listdir(dataset_folder)
    all_files.sort()
    csv_files = []

    for file in all_files:
        if not file.endswith(".csv"):
            continue
        csv_files.append(file)
        
    subcat2names = {}

    for file in all_files:
        with open(file, encoding='utf-8', errors='ignore') as csvfile: # ignore problems with strange characters
            reader = csv.reader((x.replace('\0', '') for x in csvfile), delimiter=",")
            for row in reader:
                if len(row) > Tags.SUB.value and row[Tags.SUB_ID.value]:
                    if not int(row[Tags.SUB_ID.value]) in subcat2names:
                        subcat2names[int(row[Tags.SUB_ID.value])] = row[Tags.SUB.value]

    with open('subcat2names.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for (a, b) in subcat2names.items():
            spamwriter.writerow([a, b])


if __name__ == "__main__":
    # accept key parameter as args
    parser = argparse.ArgumentParser(description='Creates a csv file with the link between the subcategories IDs and their names')
    parser.add_argument('--folder', help='the folder in which to find the csv files from which to extract the subcategories, default is "."')
    args = vars(parser.parse_args())
    
    if args['folder']:
        folder = args['folder']
        generate_subcat2names(folder)
    else:
        generate_subcat2names()