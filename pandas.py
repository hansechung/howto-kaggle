"""
Created on Mon Jan 28 11:00:45 2019

@author: Hanse
"""

"""
Created on Wed Dec 26 12:09:20 2018

@author: Hanse
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
## %matplotlib inline


DRIVE = 'C:'
USERHC = 'Users'
HANSE = 'Hanse'
MY_CODE = 'My Code'
KG_FOLDER = '2019 Kaggle How To'
ALL = 'all'
DATA_FOLDER = os.path.join(DRIVE + os.sep, USERHC, HANSE, MY_CODE, KG_FOLDER, ALL)
print(DATA_FOLDER)

transactions = pd.read_csv(os.path.join(DATA_FOLDER, 'sales_train.csv.gz'))
items = pd.read_csv(os.path.join(DATA_FOLDER, 'items.csv'))
item_categories = pd.read_csv(os.path.join(DATA_FOLDER, 'item_categories.csv'))
shops = pd.read_csv(os.path.join(DATA_FOLDER, 'shops.csv'))

print(transactions.shape)
print(items.shape)
print(item_categories.shape)
print(shops.shape)

print(transactions.head(10))
print(items.head(10))
print(item_categories.head(10))
print(shops.head(10))

# Q1: What was the maximum total revenue among all the shops in September, 2014?
# Hereinafter revenue refers to total sales minus value of goods returned.
# Hints:
# Sometimes items are returned, find such examples in the dataset.
# It is handy to split date field into [day, month, year] components 
# and use df.year == 14 and df.month == 9 in order to select target subset of dates.
# You may work with date feature as with srings, or you may first convert it to 
# pd.datetime type with pd.to_datetime function, but do not forget to set correct format argument.


def dayval(datestr):
    return int(datestr[:2])

def monthval(datestr):
    return int(datestr[3:5])

def yearval(datestr):
    return int(datestr[6:10])

def monthyear(datestr):
    return datestr[3:10]

def fxy(x, y):
    return x * y


transactions['day']=transactions.apply(lambda mx: dayval(mx.date), axis=1)
transactions['month']=transactions.apply(lambda mx: monthval(mx.date), axis=1)
transactions['year']=transactions.apply(lambda mx: yearval(mx.date), axis=1)

# transactions['month.year']=transactions.apply(lambda mx: monthyear(mx['date']), axis=1)
transactions['item_rev']=transactions.apply(lambda mx: fxy(mx.item_price, mx.item_cnt_day), axis=1)

#tr_201409 = transactions[transactions['date'][3:10]=='09.2014']

tr_201409 = transactions[(transactions.month == 9) & (transactions.year == 2014)]

rev_201409 = tr_201409.groupby(['shop_id'], as_index = False).agg({'item_rev':np.sum})

print("Q1: ", rev_201409.item_rev.max())


###############################################################################
# Q2: What item category generated the highest revenue in summer 2014?
# Submit id of the category found.
# Here we call "summer" the period from June to August.
#
# Hints:
# Note, that for an object x of type pd.Series: x.argmax() returns index of the maximum element. pd.Series can have non-trivial index (not [1, 2, 3, ... ]).
###############################################################################

def is_summer(datestr):
    if int(datestr[3:5]) >= 6 and int(datestr[3:5]) <= 8:
        return True
    else:
        return False

transactions['summer']=transactions.apply(lambda mx: is_summer(mx['date']), axis=1)

tr_2014sum=transactions[(transactions['year']==2014) & (transactions['summer']==True)]
rev_2014sum=tr_2014sum.merge(items, left_on='item_id', right_on='item_id', how='left')
rev_2014sum_bycat=rev_2014sum.groupby(['item_category_id'], as_index = False).agg({'item_rev':np.sum})

print("Q2: ", rev_2014sum_bycat.idxmax())


###############################################################################
# Q3: How many items are there, such that their price stays constant (to the best of our knowledge) during the whole period of time?
# Let's assume, that the items are returned for the same price as they had been sold.

# Find mean price per each item.
# Create vector of absolute differences between mean and actual.
# If RMS of the vector equals zero, count the item

###############################################################################

sum_stats = []
unchg_stats = []

for itemid in np.unique(transactions['item_id']):
    
    subsetd = transactions[transactions['item_id'] == itemid]
    meanpx = subsetd['item_price'].mean()
    sdpx = subsetd['item_price'].std()
    maxpx = subsetd['item_price'].max()
    minpx = subsetd['item_price'].min()
    countpx = subsetd['item_price'].count()
    
    sum_stats.append([itemid, meanpx, sdpx, maxpx, minpx, countpx])
    if maxpx == minpx:
        unchg_stats.append([itemid, meanpx, sdpx, maxpx, minpx, countpx])

print("Q3: ", len(unchg_stats))


###############################################################################
# Q4: What was the variance of the number of sold items per day sequence for the shop with shop_id = 25 in December, 2014? 
# Do not count the items, that were sold but returned back later.
# 
# Fill total_num_items_sold and days arrays, and plot the sequence with the code below.
# Then compute variance. Remember, there can be differences in how you normalize variance (biased or unbiased estimate, see link).
# Compute unbiased estimate (use the right value for ddof argument in pd.var or np.var).
# If there were no sales at a given day, do not impute missing value with zero, just ignore that day
###############################################################################

shop_id = 25

s25_dec2014 = transactions[(transactions['shop_id'] == shop_id) & (transactions['month'] == 12) & (transactions['year'] == 2014)]
items_sold = s25_dec2014.groupby(['day'], as_index = False).agg({'item_cnt_day':np.sum})

net_items_sold = items_sold[items_sold['item_cnt_day'].map(lambda x: x != 0)]        

total_num_items_sold = net_items_sold['item_cnt_day']
days = net_items_sold['day']

# Plot it
plt.plot(days, total_num_items_sold)
plt.ylabel('Num items')
plt.xlabel('Day')
plt.title("Daily revenue for shop_id = 25")
plt.show()

total_num_items_sold_var = np.var(total_num_items_sold, ddof = 1)
print("Q4: ", total_num_items_sold_var)
