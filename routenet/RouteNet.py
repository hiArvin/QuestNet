import tensorflow as tf
import networkx as nx
from topology import Topology
from utils import*
from e_graph import *
from model import ComnetLayer
import itertools




def model_fn(features,labels,mode,params):
    num_links=params['num_links']
    num_paths=params['num_paths']
    num_quests=params['num_quests']
    path_dim=params['path_dim']
    link_dim=params['link_dim']
    T=params['T']
    model = ComnetLayer(num_links,num_paths,num_quests,path_dim,link_dim,T)
    model.build()

    predictions = model(features,training=mode==tf.estimator.ModeKeys.TRAIN)

    # sess=tf.Session()
    # print(sess.run(predictions))
    predictions = tf.squeeze(predictions)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode,
                                          predictions={'predictions': predictions})
    loss = tf.losses.softmax_cross_entropy(
        logits=predictions,onehot_labels=labels,reduction=tf.losses.Reduction.MEAN
    )

    # regularization_loss = sum(model.losses)
    total_loss = loss

    tf.summary.scalar('loss', loss)
    # tf.summary.scalar('regularization_loss', regularization_loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss,
            eval_metric_ops={'accuracy':tf.metrics.accuracy(labels=labels,predictions=predictions)}
        )

    assert mode == tf.estimator.ModeKeys.TRAIN

    trainables = model.variables
    grads = tf.gradients(total_loss, trainables)
    grad_var_pairs = zip(grads, trainables)

    summaries = [tf.summary.histogram(var.op.name, var) for var in trainables]
    summaries += [tf.summary.histogram(g.op.name, g) for g in grads if g is not None]

    decayed_lr = tf.train.exponential_decay(params['learning_rate'],
                                            tf.train.get_global_step(), 82000,
                                            0.8, staircase=True)
    optimizer = tf.train.AdamOptimizer(decayed_lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(grad_var_pairs,
                                             global_step=tf.train.get_global_step())

    logging_hook = tf.train.LoggingTensorHook(
        {"Training loss": loss}, every_n_iter=10)

    return tf.estimator.EstimatorSpec(mode,
                                      loss=loss,
                                      train_op=train_op,
                                      training_hooks=[logging_hook]
                                      )

def parse_fn(example_proto):
    features = {
              'traffic': tf.VarLenFeature(tf.float32),
              'link_capacity':  tf.VarLenFeature(tf.float32),
              'labels': tf.VarLenFeature( tf.int64),
              'links': tf.VarLenFeature( tf.int64),
              'paths': tf.VarLenFeature(tf.int64),
              'sequences':tf.VarLenFeature(tf.int64)
        }
    tf_records = tf.io.parse_single_example(example_proto, features)
    feed_dict={
        'traffic':tf.sparse.to_dense(tf_records['traffic']),
        'link_capacity': tf.sparse.to_dense(tf_records['link_capacity']),
        'links': tf.sparse.to_dense(tf_records['links']),
        'paths': tf.sparse.to_dense(tf_records['paths']),
        'sequences': tf.sparse.to_dense(tf_records['sequences'])
    }
    labels=tf.sparse.to_dense(tf_records['labels'])
    labels=tf.one_hot(labels,depth=3)

    return feed_dict,labels

@tf.function
def input_fn(filename,mode=tf.estimator.ModeKeys.TRAIN,num_epochs=None,batch_size=32):
    data=tf.data.TFRecordDataset([filename])
    dataset = data.map(lambda buf:parse_fn(buf))
    dataset = dataset.shuffle(buffer_size=10000)
    # dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)

    return dataset


topo=Topology(num_core=1,num_converge=1,num_access=1)
graph = topo.graph
num_edges = nx.number_of_edges(graph)
num_nodes = nx.number_of_nodes(graph)


hparams={'num_paths':3,'num_quests':5,'num_links':num_edges,'path_dim':32,'link_dim':32,'T':4,'learning_rate':0.001}

model_dir='./logs'
train_steps=100
datafile='data/s_scale_p3q5.tfrecord'
testfile='data/s_scale_p3q5_test.tfrecord'


my_checkpointing_config = tf.estimator.RunConfig(
    save_checkpoints_secs=10*60,  # Save checkpoints every 10 minutes
    keep_checkpoint_max=20  # Retain the 10 most recent checkpoints.
)

estimator = tf.estimator.Estimator(
    model_fn = model_fn,
    model_dir=model_dir,
    params=hparams,
    config=my_checkpointing_config
    )

train_spec = tf.estimator.TrainSpec(input_fn=lambda:input_fn(datafile), max_steps=train_steps)
eval_spec = tf.estimator.EvalSpec(input_fn=lambda:input_fn(datafile),throttle_secs=10*60)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

predictions = list(itertools.islice(estimator.predict(input_fn=lambda:input_fn(datafile)),5))

print(predictions)


