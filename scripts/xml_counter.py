from xml.dom import minidom
from os import listdir
import lxml.etree

total_items = 0

#files are in the dataset folder
for file in listdir("./dataset"):
    doc = lxml.etree.parse("./dataset/" + file)
    total_items += int(doc.xpath('count(//product)'))
    print(total_items)

    #code not optimized
#     mydoc = minidom.parse("./dataset/" + file)
#     total_items += len(mydoc.getElementsByTagName('product'))