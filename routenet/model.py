import tensorflow as tf
class ComnetLayer(tf.keras.Model):
    def __init__(self, num_links, num_paths, num_quests, path_dim, link_dim, T, output_units=1):
        super(ComnetLayer, self).__init__()
        self.num_links = num_links
        self.num_paths = num_paths
        self.num_quests = num_quests
        self.total_paths = num_paths * num_quests
        self.link_dim = link_dim
        self.path_dim = path_dim
        self.T = T
        # neural network cells
        self.edge_update = tf.keras.layers.GRUCell(self.link_dim)
        self.path_update = tf.keras.layers.GRUCell(self.path_dim)

        # readout cells
        self.readout = tf.keras.models.Sequential()

        self.readout.add(tf.keras.layers.Dense(256,
                                               activation=tf.nn.selu,
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1)))
        self.readout.add(tf.keras.layers.Dropout(rate=0.5))
        self.readout.add(tf.keras.layers.Dense(256,
                                               activation=tf.nn.selu,
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1)))
        self.readout.add(tf.keras.layers.Dropout(rate=0.5))

        self.readout.add(tf.keras.layers.Dense(output_units,
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
                                               activation=tf.nn.relu))

    def build(self, input_shape=None):
        self.edge_update.build(tf.TensorShape([None, self.path_dim]))
        self.path_update.build(tf.TensorShape([None, self.link_dim]))
        self.readout.build(input_shape=[None, self.path_dim])
        self.built = True

    def call(self, inputs, training=False):
        f_ = inputs
        shape = tf.stack([self.num_links, self.link_dim - 1], axis=0)
        link_state = tf.concat([
            tf.expand_dims(f_['link_capacity'], axis=1),
            tf.zeros(shape)
        ], axis=1)
        shape = tf.stack([self.total_paths, self.path_dim - 1], axis=0)
        path_state = tf.concat([
            tf.expand_dims(f_['traffic'][0:self.total_paths], axis=1),
            tf.zeros(shape)
        ], axis=1)
        links = f_['links']
        paths = f_['paths']
        seqs = f_['sequences']

        # labels = self.placeholders['labels']
        for _ in range(self.T):
            h_tild = tf.gather(link_state, links)

            ids = tf.stack([paths, seqs], axis=1)
            max_len = tf.reduce_max(seqs) + 1
            shape = tf.stack([self.total_paths, max_len, self.link_dim])
            lens = tf.math.segment_sum(data=tf.ones_like(paths),
                                       segment_ids=paths)

            link_inputs = tf.scatter_nd(ids, h_tild, shape)
            outputs, path_state = tf.compat.v1.nn.dynamic_rnn(self.path_update,
                                                              link_inputs,
                                                              sequence_length=lens,
                                                              initial_state=path_state,
                                                              dtype=tf.float32)
            m = tf.gather_nd(outputs, ids)
            m = tf.math.unsorted_segment_sum(m, links, self.num_links)

            # Keras cell expects a list
            link_state, _ = self.edge_update(m, [link_state])

        # readout
        r = self.readout(path_state, training=True)
        # quest_shape=tf.stack([self.num_quests,self.num_paths,self.path_dim])
        # quest_state=tf.scatter_nd(self.quest_ids,path_state,quest_shape)
        return tf.reshape(r, [self.num_quests, self.num_paths])