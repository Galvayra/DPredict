
NUM_FOLDS = 1
EPOCH = 2000
IS_CLOSED = False


def show_options():
    if NUM_FOLDS > 1:
        option = str(NUM_FOLDS) + " cross validation"
    else:
        option = str(NUM_FOLDS) + " test"

    print("\n", option)
    print(" EPOCH -", EPOCH)

    if IS_CLOSED:
        print(" CLOSED DATA SET\n\n\n")
    else:
        print(" OPENED DATA SET\n\n\n")


show_options()
