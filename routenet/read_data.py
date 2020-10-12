import tensorflow as tf



# def parse_fn(example_proto):
#     features = {
#               'traffic': tf.io.VarLenFeature(tf.float32),
#               'link_capacity':  tf.io.VarLenFeature(tf.float32),
#               'labels': tf.io.VarLenFeature( tf.int64),
#               'links': tf.io.VarLenFeature( tf.int64),
#               'paths': tf.io.VarLenFeature(tf.int64),
#               'sequences':tf.io.VarLenFeature(tf.int64)
#         }
#     tf_records = tf.io.parse_single_example(example_proto, features)
#     feed_dict={
#         'traffic':tf.sparse.to_dense(tf_records['traffic']),
#         'link_capacity': tf.sparse.to_dense(tf_records['link_capacity']),
#         'links': tf.sparse.to_dense(tf_records['links']),
#         'paths': tf.sparse.to_dense(tf_records['paths']),
#         'sequences': tf.sparse.to_dense(tf_records['sequences'])
#     }
#     labels=tf.sparse.to_dense(tf_records['labels'])
#     labels=tf.one_hot(labels,depth=3)
#     print(feed_dict)
#     return feed_dict,labels
#
# @tf.function
# def input_fn(filename,mode=tf.estimator.ModeKeys.TRAIN,num_epochs=None,batch_size=32):
#     data=tf.data.TFRecordDataset([filename])
#     dataset = data.map(lambda buf:parse_fn(buf))
#     dataset = dataset.shuffle(buffer_size=10000)
#     # dataset = dataset.batch(batch_size)
#     dataset = dataset.repeat(num_epochs)
#     for d in dataset:
#         print(d)






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
        'labels': tf.sparse.to_dense(tf_record['labels']),
        'links': tf.sparse.to_dense(tf_record['links']),
        'paths': tf.sparse.to_dense(tf_record['paths']),
        'sequences': tf.sparse.to_dense(tf_record['sequences'])
    }
    labels = tf.sparse.to_dense(tf_record['labels'])
    return feed_dict, labels

parsed_dataset = raw_dataset.map(_parse_function)
parse=parsed_dataset.make_one_shot_iterator()
p=parse.get_next()
sess=tf.Session()
for i in p:
    print(sess.run(p))