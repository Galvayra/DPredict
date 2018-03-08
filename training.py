from modeling.MyOneHotEncoder import MyOneHotEncoder
from predict import logistic_regression, predict_svm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from variables import NUM_FOLDS
import numpy as np


def k_fold_cross_validation(myData, is_closed=False):
    def _set_x_dict(exception_list=list(), _is_test=False):
        x_dict = dict()

        if _is_test:
            for _k, _vector_list in myData.data_dict.items():
                x_dict[_k] = _vector_list[i * subset_size:][:subset_size]
        else:
            if is_closed:
                for _k, _vector_list in myData.data_dict.items():
                    x_dict[_k] = _vector_list
            else:
                for _k, _vector_list in myData.data_dict.items():
                    if _k in exception_list:
                        x_dict[_k] = _vector_list
                    else:
                        x_dict[_k] = _vector_list[:i * subset_size] + _vector_list[(i + 1) * subset_size:]

        return x_dict

    subset_size = int(len(myData.y_data) / NUM_FOLDS) + 1

    total_accuracy = 0
    total_score = 0

    fig = plt.figure(figsize=(10, 6))
    fig.suptitle("ROC CURVE", fontsize=16)
    svm_plot = plt.subplot2grid((2, 2), (0, 0))
    logistic_plot = plt.subplot2grid((2, 2), (0, 1))
    svm_plot.set_title("SVM")
    logistic_plot.set_title("Logistic regression")

    svm_plot.set_ylabel("TPR (sensitivity)")
    svm_plot.set_xlabel("1 - specificity")
    logistic_plot.set_ylabel("TPR (sensitivity)")
    logistic_plot.set_xlabel("1 - specificity")

    # K fold validation,  K = 10
    for i in range(NUM_FOLDS):
        print("\n\nNum Fold : %d times" % (i + 1))

        if is_closed:
            y_train = myData.y_data
        else:
            y_train = myData.y_data[:i * subset_size] + myData.y_data[(i + 1) * subset_size:]

        y_test = myData.y_data[i * subset_size:][:subset_size]

        # init MyOneHotEncoder
        myOneHotEncoder = MyOneHotEncoder()

        # set encoding original data what column class is in the exception list
        # J : 연령, AO : 수축혈압, AP : 이완혈압, AQ : 맥박수, AR : 호흡수, AS : 체온 (scalar data)
        myOneHotEncoder.encoding(_set_x_dict(exception_list=["J", "AO", "AP", "AQ", "AR", "AS"]))

        # get x_data from dictionary(data set), and set data count
        x_train = myOneHotEncoder.fit(_set_x_dict(_is_test=False), len(y_train))
        x_test = myOneHotEncoder.fit(_set_x_dict(_is_test=True), len(y_test))

        print("dims - ", len(x_train[0]))
        print("training count -", len(y_train), "\t mortality count -", myData.counting_mortality(y_train))
        print("test     count -", len(y_test), "\t mortality count -", myData.counting_mortality(y_test), "\n")

    ######Logistic Regression

        score = logistic_regression(x_train, y_train, x_test, y_test)
        logit_fpr, logit_tpr, _ = roc_curve(y_test, score)
        roc_auc = auc(logit_fpr, logit_tpr)
        logistic_plot.plot(logit_fpr, logit_tpr, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % ((i + 1), roc_auc))

    ######Logistic Regression end

    #####SVM

        accuracy, score, probas_, y_test_np = predict_svm(x_train, y_train, x_test, y_test)
        svm_fpr, svm_tpr, _ = roc_curve(y_test_np, probas_[:, 1])
        roc_auc = auc(svm_fpr, svm_tpr)

        svm_plot.plot(svm_fpr, svm_tpr, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % ((i + 1), roc_auc))

        total_accuracy += accuracy
        total_score += score

    #####SVM end

    print("Total accuracy -", total_accuracy / NUM_FOLDS)
    print("Total score -", total_score / NUM_FOLDS)
    logistic_plot.legend(loc="lower right")
    svm_plot.legend(loc="lower right")
    plt.show()


def one_fold_validation(myData, is_closed=False):
    def _set_x_dict(exception_list=list(), _is_test=False):
        x_dict = dict()

        if _is_test:
            for _k, _vector_list in myData.data_dict.items():
                x_dict[_k] = _vector_list[:subset_size]
        else:
            if is_closed:
                for _k, _vector_list in myData.data_dict.items():
                    x_dict[_k] = _vector_list
            else:
                for _k, _vector_list in myData.data_dict.items():
                    if _k in exception_list:
                        x_dict[_k] = _vector_list
                    else:
                        x_dict[_k] = _vector_list[subset_size:]


        return x_dict

    subset_size = int(len(myData.y_data) / NUM_FOLDS) + 1

    total_accuracy = 0
    total_score = 0

    fig = plt.figure(figsize=(10, 6))
    fig.suptitle("ROC CURVE", fontsize=16)
    svm_plot = plt.subplot2grid((2, 2), (0, 0))
    logistic_plot = plt.subplot2grid((2, 2), (0, 1))
    svm_plot.set_title("SVM")
    logistic_plot.set_title("Logistic regression")

    svm_plot.set_ylabel("TPR (sensitivity)")
    svm_plot.set_xlabel("1 - specificity")
    logistic_plot.set_ylabel("TPR (sensitivity)")
    logistic_plot.set_xlabel("1 - specificity")

    subset_size = int(len(myData.y_data) / 10)

    if is_closed:
        y_train = myData.y_data
    else:
        y_train = myData.y_data[subset_size:]

    y_test = myData.y_data[:subset_size]

    # init MyOneHotEncoder
    myOneHotEncoder = MyOneHotEncoder()

    # set encoding original data what column class is in the exception list
    # J : 연령, AO : 수축혈압, AP : 이완혈압, AQ : 맥박수, AR : 호흡수, AS : 체온 (scalar data)
    myOneHotEncoder.encoding(_set_x_dict(exception_list=["J", "AO", "AP", "AQ", "AR", "AS"]))

    # get x_data from dictionary(data set), and set data count
    x_train = myOneHotEncoder.fit(_set_x_dict(), len(y_train))
    x_test = myOneHotEncoder.fit(_set_x_dict(_is_test=True), len(y_test))

    print("dims - ", len(x_train[0]))
    print("training count -", len(y_train), "\t mortality count -", myData.counting_mortality(y_train))
    print("test     count -", len(y_test), "\t mortality count -", myData.counting_mortality(y_test), "\n")

    # x_train_np = np.array([np.array(j) for j in x_train])
    # x_test_np = np.array([np.array(j) for j in x_test])
    # y_train_np = np.array([np.array(j) for j in y_train])
    # y_test_np = np.array([np.array(j) for j in y_test])
    #
    # print(np.shape(x_train_np), np.shape(y_train_np))
    # print(np.shape(x_test_np), np.shape(y_test_np))

    #####Logistic Regression

    score = logistic_regression(x_train, y_train, x_test, y_test)
    logit_fpr, logit_tpr, _ = roc_curve(y_test, score)
    roc_auc = auc(logit_fpr, logit_tpr)
    logistic_plot.plot(logit_fpr, logit_tpr, alpha=0.3, label='ROC fold 1 (AUC = %0.2f)' % roc_auc)

    ######Logistic Regression end

    #####SVM

    accuracy, score, probas_, y_test_np = predict_svm(x_train, y_train, x_test, y_test)
    svm_fpr, svm_tpr, _ = roc_curve(y_test_np, probas_[:, 1])
    roc_auc = auc(svm_fpr, svm_tpr)

    svm_plot.plot(svm_fpr, svm_tpr, alpha=0.3, label='ROC fold 1 (AUC = %0.2f)' % roc_auc)

    total_accuracy += accuracy
    total_score += score

    #####SVM end

    print("Total accuracy -", total_accuracy / NUM_FOLDS)
    print("Total score -", total_score / NUM_FOLDS)
    logistic_plot.legend(loc="lower right")
    svm_plot.legend(loc="lower right")
    plt.show()
