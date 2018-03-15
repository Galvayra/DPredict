# -*- coding: utf-8 -*-
import time
import pickle
from variables import NUM_FOLDS, IS_CLOSED
from training import *

start_time = time.time()

DIR_NAME = "Pickle/"
FILE_NAME = "data.p"


if __name__ == '__main__':

    try:
        with open(DIR_NAME + FILE_NAME, 'rb') as file:
            myData = pickle.load(file)
    except FileNotFoundError:
        print("\nPlease execute encoding script !")
    else:
        if IS_CLOSED:
            closed_validation(myData)
        else:
            if NUM_FOLDS > 1:
                k_fold_cross_validation(myData)
            else:
                one_fold_validation(myData)

        end_time = time.time()
        print("processing time     --- %s seconds ---" % (time.time() - start_time))

