
NUM_FOLDS = 5
# using test set 10 %
RATIO = 10
IS_CLOSED = True


def show_options():
    if IS_CLOSED:
        print("\n\n========== CLOSED DATA SET ==========\n")
    else:
        print("\n\n========== OPENED DATA SET ==========\n")

        if NUM_FOLDS > 1:
            option = str(NUM_FOLDS) + " cross validation"
        else:
            option = str(NUM_FOLDS) + " test, Ratio - " + str(RATIO)

        print(option)


show_options()
