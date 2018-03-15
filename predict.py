import tensorflow as tf
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, accuracy_score
from variables import NUM_FOLDS, IS_CLOSED

HIDDEN_LAYER = 800
EPOCH = 300


def logistic_regression(x_train, y_train, x_test, y_test):
    data_count = len(x_train)
    predict_count = len(y_test)
    dimension = len(x_train[0])

    X = tf.placeholder(dtype=tf.float32, shape=[None, dimension])
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    W = tf.get_variable("weight1", dtype=tf.float32, shape=[dimension, HIDDEN_LAYER],
                        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.random_normal([HIDDEN_LAYER]), name="bias1")
    L = tf.nn.relu(tf.matmul(X, W) + b)

    W2 = tf.get_variable("weight2", dtype=tf.float32, shape=[HIDDEN_LAYER, 1],
                        initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([1]), name="bias2")
    hypothesis = tf.sigmoid(tf.matmul(L, W2) + b2)

    with tf.name_scope("cost"):
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

        h, p, a = sess.run([hypothesis, predict, accuracy], feed_dict={X: x_test, Y: y_test})

    if NUM_FOLDS == 1 or IS_CLOSED:
        print("\n\nsave log!\n")

    precision = precision_score(y_test, p)
    recall = recall_score(y_test, p)
    accuracy = accuracy_score(y_test, p)

    print('logistic regression')
    print('Precision : %.2f' % (precision*100))
    print('Recall : %.2f' % (recall*100))
    print('Accuracy : %.2f' % (accuracy*100))

    tf.reset_default_graph()

    return h, accuracy, precision, recall


def predict_svm(x_train, y_train, x_test, y_test):

    model = SVC(kernel='rbf', C=1.0, random_state=None, probability=True)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # y_score = model.decision_function(x_test)
    probas_ = model.predict_proba(x_test)

    # average = average_precision_score(y_test_np, y_score)

    # precision, recall, _ = precision_recall_curve(y_test_np, y_score)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    # print(y_pred)

    print('SVM')
    print('Precision : %.2f' % (precision*100))
    print('Recall : %.2f' % (recall*100))
    print('Accuracy : %.2f' % (accuracy*100))

    return probas_, accuracy, precision, recall

