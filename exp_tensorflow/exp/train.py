from __future__ import division
from __future__ import print_function
import time
import tensorflow as tf
from utils import *
from metrics import *
from models import GCN, MLP, Dense
from scipy.stats import zscore
import pickle as pkl
import numpy as np


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2')
flags.DEFINE_float('dropout', 0.3, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).') # only for Tensorflow
flags.DEFINE_float('alpha', 0.5, 'alpha')
flags.DEFINE_string('struc_features', 'all', 'Measurement of node importance')  # 'eigen', 'pagerank', 'degree', 'closeness', 'all'
flags.DEFINE_integer('n_train', 10, 'Number of examples per class to train')
flags.DEFINE_integer('n_val', 500, 'Number of examples per class to train')
flags.DEFINE_integer('n_test', 1000, 'Number of examples per class to train')
flags.DEFINE_integer('seed', 100, 'random seed')

# Set random seed
seed = FLAGS.seed
np.random.seed(seed)
tf.set_random_seed(seed)

# Load data
adj, features, labels = tuple(pkl.load(open('../data/' + FLAGS.dataset + '_data.pkl', 'rb')))
input_shape = features.shape
features = preprocess_features(features)
struc_feat = np.load('../data/' + FLAGS.dataset + '_' + FLAGS.struc_features + '.npy')
struc_feat = zscore(struc_feat, axis=0)

support = [preprocess_adj(adj)]
num_supports = 1
model_func = GCN

train_mask, test_mask, val_mask, y_train, y_test, y_val = split_data(labels, FLAGS.n_train, FLAGS.n_test, FLAGS.n_val)

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
    'tRU': tf.placeholder(tf.float32, shape=())
}

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

model = model_func(placeholders, input_dim=features[2][1], logging=True)

aux_layer = Dense(FLAGS.hidden1, struc_feat.shape[1], placeholders=placeholders, act=lambda x:x)
aux_act = aux_layer(model.activations[-2])

rloss = FLAGS.alpha * tf.reduce_mean(tf.square(aux_act - struc_feat))
model.loss += rloss

model.opt_op = model.optimizer.minimize(model.loss)

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):
    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

   # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

