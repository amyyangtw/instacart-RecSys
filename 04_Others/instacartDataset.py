import os
import csv
import sys
import re
import numpy as np

from surprise import Dataset
from surprise import Reader

from collections import defaultdict


class InstacartDataset:

    itemID_to_name = {}
    name_to_itemID = {}
    reordersPath = '../../dataset/reorder_100k.csv'
    itemPath = '../../dataset/products.csv'

    def loadReordersDataset(self):
        reordersDataset = 0
        self.itemID_to_name = {}
        self.name_to_itemID = {}

        reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
        reordersDataset = Dataset.load_from_file(self.reordersPath, reader=reader)

        with open(self.itemPath, newline='', encoding='utf-8') as csvfile:
                itemReader = csv.reader(csvfile)
                next(itemReader)  #Skip header line
                for row in itemReader:
                    itemID = int(row[0])
                    itemName = row[1]
                    self.itemID_to_name[itemID] = itemName
                    self.name_to_itemID[itemName] = itemID

        return reordersDataset
    
    def getItemName(self, item_id):
        if self.itemID_to_name.get(item_id):
            return self.itemID_to_name[item_id]
        else:
            return 'item not found'
        