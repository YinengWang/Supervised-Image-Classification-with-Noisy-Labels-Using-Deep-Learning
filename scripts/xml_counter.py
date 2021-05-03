""""
    This is a short script to count the products in the dataset folder
"""

from xml.dom import minidom
from os import listdir
import lxml.etree

total_items = 0.0


files = listdir("./dataset")

#files are in the dataset folder
for file in files:

    print("Parsing " + file)

    doc = lxml.etree.parse("./dataset/" + file)
    count = doc.xpath('count(//product)')
    total_items += count
    del doc
    del count
    print(total_items)
    print("\n")