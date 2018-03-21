import tensorflow as tf
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, accuracy_score
from variables import NUM_FOLDS, IS_CLOSED

NUM_HIDDEN_LAYER = 0
NUM_HIDDEN_DIMENSION = 0
EPOCH = 400


def logistic_regression(x_train, y_train, x_test, y_test, fold):
    data_count = len(x_train)
    predict_count = len(y_test)
    num_input_node = len(x_train[0])

    if NUM_HIDDEN_DIMENSION:
        num_hidden_node = NUM_HIDDEN_DIMENSION
    else:
        num_hidden_node = len(x_train[0])

    tf_x = tf.placeholder(dtype=tf.float32, shape=[None, num_input_node])
    tf_y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    tf_weight = list()
    tf_bias = list()
    tf_layer = [tf_x]

    print("\nepoch -", EPOCH)
    if NUM_HIDDEN_LAYER:
        print("num of  hidden layers  -", NUM_HIDDEN_LAYER)
        print("num of nodes in hidden -", num_hidden_node, "\n\n")

    for i in range(NUM_HIDDEN_LAYER):
        if i == 0:
            tf_weight.append(tf.get_variable("h_weight_" + str(i + 1), dtype=tf.float32,
                                             shape=[num_input_node, num_hidden_node],
                                             initializer=tf.contrib.layers.xavier_initializer()))
        else:
            tf_weight.append(tf.get_variable("h_weight_" + str(i + 1), dtype=tf.float32,
                                             shape=[num_hidden_node, num_hidden_node],
                                             initializer=tf.contrib.layers.xavier_initializer()))
        tf_bias.append(tf.Variable(tf.random_normal([num_hidden_node]), name="h_bias_" + str(i + 1)))
        tf_layer.append(tf.nn.relu(tf.matmul(tf_layer[i], tf_weight[i]) + tf_bias[i]))

    tf_weight.append(tf.get_variable("o_weight", dtype=tf.float32, shape=[num_hidden_node, 1],
                                     initializer=tf.contrib.layers.xavier_initializer()))
    tf_bias.append(tf.Variable(tf.random_normal([1]), name="o_bias"))

    hypothesis = tf.sigmoid(tf.matmul(tf_layer[-1], tf_weight[-1]) + tf_bias[-1])

    with tf.name_scope("cost"):
        cost = -tf.reduce_mean(tf_y * tf.log(hypothesis) + (1-tf_y) * tf.log(1-hypothesis))

        if NUM_FOLDS == 1 or IS_CLOSED:
            cost_summ = tf.summary.scalar("cost", cost)

    train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

    #cut off
    predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, tf_y), dtype=tf.float32))

    if NUM_FOLDS == 1 or IS_CLOSED:
        accuracy_summ = tf.summary.scalar("accuracy", accuracy)

    with tf.Session() as sess:
        merged_summary = tf.summary.merge_all()
        if NUM_FOLDS == 1 or IS_CLOSED:
            writer = tf.summary.FileWriter("./logs/log_0" + str(fold + 1))
            writer.add_graph(sess.graph)  # Show the graph

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if NUM_FOLDS == 1 or IS_CLOSED:
            for step in range(EPOCH + 1):
                summary, cost_val, _ = sess.run([merged_summary, cost, train], feed_dict={tf_x: x_train, tf_y: y_train})
                writer.add_summary(summary, global_step=step)

                if step % (EPOCH / 10) == 0:
                    print(str(step).rjust(6), cost_val)
        else:
            for step in range(EPOCH + 1):
                cost_val, _ = sess.run([cost, train], feed_dict={tf_x: x_train, tf_y: y_train})

                if step % (EPOCH / 10) == 0:
                    print(str(step).rjust(6), cost_val)

        h, p, a = sess.run([hypothesis, predict, accuracy], feed_dict={tf_x: x_test, tf_y: y_test})

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

    model = SVC(kernel='linear', C=1.0, random_state=None, probability=True)
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

