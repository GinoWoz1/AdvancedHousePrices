# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 12:02:54 2018

@author: jjonus
"""

import itertools


features = ['Foundation_PConc','YearBuilt','YearRemodAdd','Exterior1st_VinylSd','KitchenQual_Gd','ExterQual_Gd','BsmtQual_Gd']

list_o_lists = []
for c in range(1,len(features)+1):
    print(c)
    data =  list(itertools.combinations(features,c))
    list_o_lists.append(data)
new_list = [[]]
for i in list(list_o_lists):
    for c in i:
        c = c.replace(")","]").replace(",]","]")
        new_list.append(c)

