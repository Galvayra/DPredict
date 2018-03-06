import tensorflow as tf
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

EPOCH = 5000


def logistic_regression(x_train, y_train, x_test, y_test):
    data_count = len(x_train)
    predict_count = len(y_test)
    dimension = len(x_train[0])

    X = tf.placeholder(dtype=tf.float32, shape=[None, dimension])
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    # W1 = tf.Variable(tf.random_normal([dimension, 10]), name="weight1")
    # b1 = tf.Variable(tf.random_normal([10]), name="bias1")
    #
    # W2 = tf.Variable(tf.random_normal([10, 10]), name="weight2")
    # b2 = tf.Variable(tf.random_normal([10]), name="bias2")
    #
    # W3 = tf.Variable(tf.random_normal([10, 1]), name="weight3")
    # b3 = tf.Variable(tf.random_normal([1]), name="bias3")
    #
    # layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
    # layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)
    # hypothesis = tf.sigmoid(tf.matmul(layer2, W3) + b3)

    W = tf.Variable(tf.random_normal([dimension, 1]), name="weight")
    b = tf.Variable(tf.random_normal([1]), name="bias")
    hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))

    train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

    #cut off
    predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, Y), dtype=tf.float32))

    # print("\n\nTraining Matrix size -", len(x_train), "*", dimension)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for step in range(EPOCH + 1):
            cost_val, _ = sess.run([cost, train], feed_dict={X: x_train, Y: y_train})

            if step % (EPOCH / 10) == 0:
                print(step, cost_val)

        h, c, a = sess.run([hypothesis, predict, accuracy], feed_dict={X: x_test, Y: y_test})

        # print("\nAccuracy: ", a)

    return h


def predict_svm(x_train, y_train, x_test, y_test):

    # train_count=0
    # test_count=0
    #
    # for train in y_train:
    #     if train[0] == 0:
    #         train_count += 1
    # print("percentage : %.2f, count : %d, one_count : %d" % ((train_count/len(y_train)*100), train_count, len(y_train)-train_count))
    #
    # for test in y_test:
    #     if test[0] == 0:
    #         test_count += 1
    # print("percentage : %.2f, count : %d, one_count : %d\n" % ((test_count/len(y_test)*100), test_count, len(y_test)-test_count))

    x_train_np = np.array([np.array(j) for j in x_train])
    x_test_np = np.array([np.array(j) for j in x_test])
    y_train_np = np.array([np.array(j) for j in y_train])
    y_test_np = np.array([np.array(j) for j in y_test])

    # sc = StandardScaler()5
    # sc.fit(x_train,y_train)
    # x_train_std= sc.transform(x_train)
    # x_test_std= sc.transform(x_test)
    #
    model = SVC(kernel='linear', C=1.0, random_state=0, probability=True)
    model.fit(x_train_np, y_train_np)
    y_pred = model.predict(x_test_np)
    y_score = model.decision_function(x_test_np)
    probas_ = model.predict_proba(x_test_np)

    average = average_precision_score(y_test, y_score)

    precision, recall, _ = precision_recall_curve(y_test_np, y_score)

    print('Accuracy: %.2f' % (accuracy_score(y_test_np, y_pred)*100))
    print('Average precision-recall : %.2f' % average)

    return accuracy_score(y_test_np, y_pred), average, probas_, y_test_np