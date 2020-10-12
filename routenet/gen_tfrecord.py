import tensorflow as tf
from topology import Topology
import networkx as nx
from e_graph import *
from routenet_labelling import gen_label
from utils import *

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_features(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



topo=Topology(num_core=1,num_access=1,num_converge=1)
graph=topo.graph

num_edges = nx.number_of_edges(graph)
num_nodes = nx.number_of_nodes(graph)

num_paths=3
num_quests=5

_, init_link_capacity = nodeGraph_to_edgeGraph(graph, support=True)
flow_generator = topo.gen_flows()


tf_file='./data/s_scale_p3q5.tfrecord'
writer = tf.python_io.TFRecordWriter(tf_file)

num_data=100
link_capacity=init_link_capacity



for i in range(num_data):
    if (i+1)%100 ==0:
        link_capacity = init_link_capacity
    flow = []
    for i in range(num_quests):
        flow.append(next(flow_generator))
    paths, ids, seqs, labels, occupy, traffic = gen_label(topo.graph, flow, link_capacity, num_paths=num_paths,
                                                          num_quests=num_quests)

    example = tf.train.Example(features=tf.train.Features(feature={
        'traffic': _float_features(normalization(traffic)),
        'link_capacity': _float_features(normalization(link_capacity)),
        'labels':_int64_features(labels),
        'links': _int64_features(paths),
        'paths': _int64_features(ids),
        'sequences': _int64_features(seqs)
    }
    ))
    link_capacity=update_capacity(link_capacity,occupy)
    writer.write(example.SerializeToString())
writer.close()
