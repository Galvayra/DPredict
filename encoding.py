# -*- coding: utf-8 -*-

import sys
from os import path

try:
    import DPredict
except ImportError:
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from DPredict.dataset.dataHandler import DataHandler
from DPredict.modeling.vectorization import MyVector


if __name__ == '__main__':
    myData = DataHandler()
    myData.set_labels()
    myData.free()

    # myData.show_data()
    myVector = MyVector(myData)
    myVector.encoding()
    myVector.dump()
