import pandas as pd
import numpy as np
import math
import copy
import random


def remove_nan(lists):
    newlists = []
    for tr in lists:
        newlists.append([int(x) for x in tr if str(x) != 'nan'])
    return(newlists)

def import_log(filepath):
    df = pd.read_csv(filepath)
    return(remove_nan(df.values.tolist()))

def save_log(loglist, filename): #save a list of lists 
    df = pd.DataFrame.from_records(loglist)
    df.to_csv(filename, index=False)
    
def return_train_test(log, variants): #remove the variants from a log and return new log
    return([trace for trace in log if trace not in variants], [trace for trace in log if trace in variants])

def count_variant(log, variant): #count how many times a variant comes up in list
    c = 0
    for trace in log:
        if trace == variant:
            c += 1
    return(c)

def get_counts(log, variants):
    counts = []
    for var in variants:
        counts.append(count_variant(log, var))
    return counts

def get_splits(log_locations, variants_location):
    full_log = import_log(log_locations)
    variants = import_log(variants_location)
    percentage = math.floor(len(variants) * 0.1)
    test_and_val_variants_i = random.sample(range(0, len(variants)), percentage * 2)
    test_variants_i = test_and_val_variants_i[0:percentage]
    test_variants = []
    for i in range(0, percentage):
        test_variants.append(variants[test_variants_i[i]])
    tr, te = return_train_test(full_log, test_variants)
    tr_cop = copy.deepcopy(tr)  
    save_log(tr, 'Full_Train.csv')
    save_log(te, 'Test.csv')
    val_variants_i = test_and_val_variants_i[percentage:(2*percentage)]
    val_variants = []
    for i in range(0, percentage):
        val_variants.append(variants[val_variants_i[i]])
    VBR_tr, VBR_val = return_train_test(tr_cop, val_variants) 
    save_log(VBR_tr, 'VBR_Tr.csv')
    save_log(VBR_val, 'VBR_Val.csv')
    
    where_split = int(len(tr) * 0.90) 
    save_log(tr[0:where_split], 'RAND_Tr.csv')
    save_log(tr[where_split:-1], 'RAND_Val.csv')
    
    save_log(VBR_tr[0:where_split], 'RVBR_Tr.csv')
    save_log(VBR_tr[where_split:-1]+VBR_val, 'RVBR_Val.csv')
    
    