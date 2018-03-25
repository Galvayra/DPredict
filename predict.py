# -*- coding: utf-8 -*-
import sys
import json
from os import path

try:
    import DPredict
except ImportError:
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from DPredict.dataset.variables import DATA_READ
from DPredict.modeling.variables import *
from DPredict.learning.train import MyTrain
import DPredict.arguments as op


if __name__ == '__main__':
    csv_name = DATA_READ.split('.')[0]

    file_name = DUMP_PATH + DUMP_FILE + "_" + csv_name + "_opened_" + str(op.NUM_FOLDS)

    try:
        with open(file_name, 'r') as file:
            vector_list = json.load(file)
    except FileNotFoundError:
        print("\nPlease execute encoding script !")
        print("make sure whether vector file is existed in", DUMP_PATH, "directory")
    else:
        print("\nRead vectors -", file_name)
        op.show_options()

        train = MyTrain(vector_list)
        train.training(do_show=op.DO_SHOW)
