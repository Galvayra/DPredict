import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)


def get_arguments():
    parser.add_argument("-epoch", "--epoch", help="set epoch for neural network (default is 2000)"
                                                  "\nyou have to use this option more than 100"
                                                  "\nUseAge : python encoding.py -epoch \n\n")
    parser.add_argument("-hidden", "--hidden", help="set a number of hidden layer (default is 0)"
                                                    "\ndefault is not using hidden layer for linear model"
                                                    "\nUseAge : python encoding.py -hidden 5 (non-linear)\n\n")
    parser.add_argument("-show", "--show", help="show plot (default is 0)"
                                                "\nUseAge : python encoding.py -show 1 (True)\n\n")
    _args = parser.parse_args()

    return _args


NUM_FOLDS = 5
RATIO = 10

args = get_arguments()

if not args.show:
    DO_SHOW = 0
else:
    try:
        DO_SHOW = int(args.show)
    except ValueError:
        print("\nInput Error type of show option!\n")
        exit(-1)
    else:
        if DO_SHOW != 1 and DO_SHOW != 0:
            print("\nInput Error show option!\n")
            exit(-1)

if not args.epoch:
    EPOCH = 2000
else:
    try:
        EPOCH = int(args.epoch)
    except ValueError:
        print("\nInput Error type of epoch option!\n")
        exit(-1)
    else:
        if EPOCH < 100:
            print("\nInput Error epoch option!\n")
            exit(-1)

if not args.hidden:
    NUM_HIDDEN_LAYER = 0
else:
    try:
        NUM_HIDDEN_LAYER = int(args.hidden)
    except ValueError:
        print("\nInput Error type of hidden option!\n")
        exit(-1)
    else:
        if NUM_HIDDEN_LAYER < 0:
            print("\nInput Error hidden option!\n")
            exit(-1)

NUM_HIDDEN_DIMENSION = 0


def show_options():
    print("\n\n========== OPENED DATA SET ==========\n")
    print("k fold -", NUM_FOLDS)
    if NUM_FOLDS == 1:
        print("test ratio -", str(RATIO) + "%")

    print("num of hidden layers -", NUM_HIDDEN_LAYER)

    print("EPOCH -", EPOCH)
