from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np

# Parameters
num_steps = 50
num_classes = 5
num_features = 14
num_trees = 10
max_nodes = 5000

# Import data
data = pd.read_csv("../5. data set/input.csv")

# Extract features and targets to numpy
input_x = data.iloc[:, :num_features].values
input_y = data.iloc[:, num_features].values

# Split train data and tes tdata
X_train, X_test, y_train, y_test = train_test_split(
    input_x,
    input_y,
    test_size=0.3, random_state=0)

# Stratified K Fold for classification
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# Input and Target placeholders
X = tf.placeholder(dtype=tf.float32, shape=[None, num_features])
Y = tf.placeholder(dtype=tf.int64, shape=[None])

# Random Forest Parameters
hparams = tensor_forest.ForestHParams(
    num_classes=num_classes,
    num_features=num_features,
    num_trees=num_trees,
    max_nodes=max_nodes).fill()

# Build the Random Forest
forest_graph = tensor_forest.RandomForestGraphs(hparams)

# Get training graph and loss
train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)

# Measure the accuracy
infer_op, _, _ = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize the variables and forest resources
init_vars = tf.group(tf.global_variables_initializer(), resources.initialize_resources(resources.shared_resources()))


def run_train(session, train_X, train_y, count):
    print("\nStart training")
    session.run(init_vars)
    acc_train_result = list()
    acc_test_result = list()
    for i in range(1, num_steps + 1):
        _, l = session.run([train_op, loss_op], feed_dict={X: train_X, Y: train_y})
        acc_train = session.run(accuracy_op, feed_dict={X: train_X, Y: train_y})
        acc_test = session.run(accuracy_op, feed_dict={X: X_test, Y: y_test})
        acc_train_result.append(acc_train)
        acc_test_result.append(acc_test)

        if i % 10 == 0 or i == 1:
            print('Step %i, Loss: %f, Acc: %f' % (i, l, acc_train))
            print('Test Accuracy : ', acc_test)
            tf.summary.FileWriter("/tmp/test_logs", session.graph)
    plt.plot(acc_train_result, label="Train")
    plt.plot(acc_test_result, label="Test")
    acc_train_chart = pd.DataFrame(acc_train_result, columns=["acc_train"])
    acc_test_chart = pd.DataFrame(acc_test_result, columns=["acc_test"])
    pd.concat([acc_train_chart, acc_test_chart], axis=1).to_csv("../2. acc graph : chart/acc_chart_"+str(count)+".csv", index=False)
    plt.legend()
    plt.savefig("../2. acc graph : chart/acc_graph_"+str(count)+".jpg")
    plt.show()


# Save session to select the best model
saver = tf.train.Saver()


def cross_validate(session):
    results = []
    filenum = 0
    crossval_count = 0
    for train_idx, val_idx in skf.split(X_train, y_train):
        train_x = X_train[train_idx]
        train_y = y_train[train_idx]
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]
        run_train(session, train_x, train_y, filenum)
        results.append(session.run(accuracy_op, feed_dict={X: X_val, Y: y_val}))
        if crossval_count == 0:
            saver.save(session, 'best result')
            save_path = saver.save(session, "bestmodel.ckpt")
            print("Model saved in path: %s" % save_path)
            crossval_count += 1
        elif crossval_count > 0:
            if results[crossval_count-1] < results[crossval_count]:
                saver.save(session, 'best result')
                save_path = saver.save(session, "bestmodel.ckpt")
                print("Model saved in path: %s" % save_path)
                crossval_count += 1
        session.close()
        session = tf.Session()
        filenum += 1
    return results


session = tf.Session()
result = cross_validate(session)
print("Cross-validation result: ", result)
plt.xticks(np.arange(0, 5, 1.0))
plt.title("5 Fold Cross Validation")
plt.plot(result)
plt.savefig("../2. acc graph : chart/5_fold_cross_validation.jpg")
plt.show()
session = tf.Session()
session.run(init_vars)
saver.restore(session, "bestmodel.ckpt")
print("Model restored.")
print("Test accuracy: ", session.run(accuracy_op, feed_dict={X: X_test, Y: y_test}))
