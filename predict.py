import tensorflow as tf
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from variables import EPOCH, NUM_FOLDS, IS_CLOSED


def logistic_regression(x_train, y_train, x_test, y_test):
    data_count = len(x_train)
    predict_count = len(y_test)
    dimension = len(x_train[0])

    X = tf.placeholder(dtype=tf.float32, shape=[None, dimension])
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    if NUM_FOLDS == 1 or IS_CLOSED:
        W = tf.get_variable("weight", shape=[dimension, 1], initializer=tf.contrib.layers.xavier_initializer())
    else:
        W = tf.Variable(tf.random_normal([dimension, 1]), name="weight")
    b = tf.Variable(tf.random_normal([1]), name="bias")
    hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

    # W3 = tf.get_variable("W3", shape=[10, 1], initializer=tf.contrib.layers.xavier_initializer())
    # b3 = tf.Variable(tf.random_normal([1]), name="bias3")
    # hypothesis = tf.sigmoid(tf.matmul(L1, W3) + b3)

    with tf.name_scope("cost") as scope:
        cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))

        if NUM_FOLDS == 1 or IS_CLOSED:
            cost_summ = tf.summary.scalar("cost", cost)

    train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

    #cut off
    predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, Y), dtype=tf.float32))

    if NUM_FOLDS == 1 or IS_CLOSED:
        accuracy_summ = tf.summary.scalar("accuracy", accuracy)

    with tf.Session() as sess:
        merged_summary = tf.summary.merge_all()
        if NUM_FOLDS == 1 or IS_CLOSED:
            writer = tf.summary.FileWriter("./logs/log_01")
            writer.add_graph(sess.graph)  # Show the graph

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if NUM_FOLDS == 1 or IS_CLOSED:
            for step in range(EPOCH + 1):
                summary, cost_val, _ = sess.run([merged_summary, cost, train], feed_dict={X: x_train, Y: y_train})
                writer.add_summary(summary, global_step=step)

                if step % (EPOCH / 10) == 0:
                    print(str(step).rjust(6), cost_val)
        else:
            for step in range(EPOCH + 1):
                cost_val, _ = sess.run([cost, train], feed_dict={X: x_train, Y: y_train})

                if step % (EPOCH / 10) == 0:
                    print(str(step).rjust(6), cost_val)

        h, c, a = sess.run([hypothesis, predict, accuracy], feed_dict={X: x_test, Y: y_test})

    if NUM_FOLDS == 1 or IS_CLOSED:
        print("\n\nsave log!\n")

    return h


def predict_svm(x_train, y_train, x_test, y_test):

    x_train_np = np.array([np.array(j) for j in x_train])
    x_test_np = np.array([np.array(j) for j in x_test])
    y_train_np = np.array([np.array(j) for j in y_train])
    y_test_np = np.array([np.array(j) for j in y_test])

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


def predict_rnn(x_train, y_train, x_test, y_test):
    pass
