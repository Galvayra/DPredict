from dataset.dataHandler import DataHandler
import pickle
import os

DIR_NAME = "Pickle/"
FILE_NAME = "data.p"


if __name__ == '__main__':
    myData = DataHandler()
    myData.set_labels()
    myData.free()

    if not os.path.isdir(DIR_NAME):
        os.mkdir(DIR_NAME)

    with open(DIR_NAME + FILE_NAME, 'wb') as file:
        pickle.dump(myData, file)
