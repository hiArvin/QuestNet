from __future__ import division
from __future__ import print_function

from model_tf2 import ComnetLayer
from topology import Topology
import tensorflow as tf
import networkx as nx

# preprocessing
topo = Topology(num_core=1, num_access=1, num_converge=1)
graph = topo.graph
num_edges = nx.number_of_edges(graph)
num_nodes = nx.number_of_nodes(graph)

# hyper-parameters
num_paths = 3
num_quests = 5
lr = 0.001
link_dim = 32
path_dim = 32


# load data
filename='data/s_scale_p3q5.tfrecord'
raw_dataset = tf.data.TFRecordDataset([filename])

def _parse_function(example_proto):
    features = {
        'traffic': tf.io.VarLenFeature(tf.float32),
        'link_capacity': tf.io.VarLenFeature(tf.float32),
        'labels': tf.io.VarLenFeature(tf.int64),
        'links': tf.io.VarLenFeature(tf.int64),
        'paths': tf.io.VarLenFeature(tf.int64),
        'sequences': tf.io.VarLenFeature(tf.int64)
    }
    tf_record=tf.io.parse_single_example(example_proto, features)
    feed_dict = {
        'traffic': tf.sparse.to_dense(tf_record['traffic']),
        'link_capacity': tf.sparse.to_dense(tf_record['link_capacity']),
        'links': tf.sparse.to_dense(tf_record['links']),
        'paths': tf.sparse.to_dense(tf_record['paths']),
        'sequences': tf.sparse.to_dense(tf_record['sequences'])
    }
    labels = tf.sparse.to_dense(tf_record['labels'])
    return (feed_dict,tf.one_hot(labels,depth=3))
parsed_dataset = raw_dataset.map(lambda buf:_parse_function(buf))

# model
model = ComnetLayer(num_links=num_edges, num_paths=num_paths, num_quests=num_quests, T=3,
                 link_dim=link_dim, path_dim=path_dim)

model.compile(optimizer='adam',loss=tf.keras.losses.binary_crossentropy)
model.fit(parsed_dataset,steps_per_epoch=20)

