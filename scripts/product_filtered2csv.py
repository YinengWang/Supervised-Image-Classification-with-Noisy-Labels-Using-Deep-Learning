""""
    This is a script to gather useful information from each product and save it in a csv file. 
    The info collected are: 
        category (from the file), name (title), thumbnail_url (thumbnail), date (release_date), img_url (big_images -> big_image), sub_category_id (catid), subcategory (categories -> category)
    The script also excludes 'bad' product recognized by having:
        <link>https://cdon.se/</link>
    It assumes that all the xml files are in the ./dataset folder.
"""

from os import listdir
import xml.etree.ElementTree as ET
import csv
import os.path

total_items = 0.0

files = listdir("./dataset")

#files are in the dataset folder
for file in files:
    if not file.endswith(".xml"):
        continue
    print("Parsing " + file)
    csv_filename = "./dataset/" + file[5: len(file) - 3] + "csv"

    if os.path.isfile(csv_filename):
       print("csv file for " + file + " already exists, skipping this one") 
       continue

    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        inside_product_tag = False
        product_is_good = True #checks if product is still sold and should be stored

        #tags to store
        tags = ['title', 'thumbnail', 'release_date', 'big_image', 'catid', 'category']
        row = {}
        for tag in tags:
            row[tag] = None
        iter = ET.iterparse("./dataset/" + file, events=("start", "end"))
        for event, elem in iter:
            if elem.tag == 'product':
                inside_product_tag = event == "start"
                if event == "end":
                    if product_is_good:
                        writer.writerow(list(row.values()))
                    for tag in tags:
                        row[tag] = None
                    elem.clear()
                else:
                    product_is_good = True

            elif elem.tag == 'link' and elem.text == "https://cdon.se/":
                product_is_good = False 
            if event == "start":
                for tag in tags:
                    if elem.tag == tag:
                        row[tag] = elem.text
                if elem.tag == "thumbnail" and not elem.text:
                    product_is_good = False
        
        del iter