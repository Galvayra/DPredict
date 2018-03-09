# -*- coding: utf-8 -*-
from dataset.dataHandler import DataHandler
import time
from variables import NUM_FOLDS, IS_CLOSED
from training import *

start_time = time.time()

if __name__ == '__main__':
    myData = DataHandler()
    myData.set_labels()
    myData.free()

    if IS_CLOSED:
        closed_validation(myData)
    else:
        if NUM_FOLDS > 1:
            k_fold_cross_validation(myData)
        else:
            one_fold_validation(myData)

    end_time = time.time()
    print("processing time     --- %s seconds ---" % (time.time() - start_time))

