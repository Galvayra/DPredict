# -*- coding: utf-8 -*-
from dataset.dataHandler import DataHandler
import time
from variables import NUM_FOLDS
from training import *

start_time = time.time()


if __name__ == '__main__':
    myData = DataHandler()
    myData.set_labels()
    myData.free()

    if NUM_FOLDS > 1:
        k_fold_cross_validation(myData, is_closed=True)
    else:
        one_fold_validation(myData, is_closed=False)

    end_time = time.time()
    print("processing time     --- %s seconds ---" % (time.time() - start_time))

