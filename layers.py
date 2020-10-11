from inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        # self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                                        name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        # if self.sparse_inputs:
        #     x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        # else:
        x = tf.nn.dropout(x, 1 - self.dropout)

        # convolve
        supports = list()
        if not self.featureless:
            pre_sup = dot(x, self.vars['weights'])
        else:
            pre_sup = self.vars['weights']
        support = dot(self.support, pre_sup, sparse=False)
        supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']
        self.outs=self.act(output)
        return self.act(output)


class PathEmbedding(Layer):
    def __init__(self,num_quests,num_paths, link_state_dim,path_state_dim, placeholders,act=tf.nn.relu, **kwargs):
        super(PathEmbedding, self).__init__(**kwargs)
        self.num_quests=num_quests
        self.num_paths=num_paths
        self.paths = placeholders['paths']
        self.idx = placeholders['index']
        self.seqs = placeholders['sequences']
        self.features=placeholders['features']
        self.flow_size=tf.reshape(placeholders['flow_size'],[num_quests*num_paths,1])
        self.link_state_dim = link_state_dim
        self.path_update = tf.keras.layers.GRUCell(path_state_dim)
        self.act=act



    def _call(self, inputs):
        inputs= tf.math.l2_normalize(inputs)
        self.path_update.build(tf.TensorShape([None, self.link_state_dim]))

        h_tild = tf.gather(inputs, self.paths)

        ids = tf.stack([self.idx, self.seqs], axis=1)
        max_len = tf.reduce_max(self.seqs) + 1
        shape = tf.stack([self.num_quests*self.num_paths, max_len, self.link_state_dim])
        lens = tf.math.segment_sum(data=tf.ones_like(self.idx),
                                   segment_ids=self.idx)
        link_inputs = tf.scatter_nd(ids, h_tild, shape)
        # initial_state =tf.zeros([self.num_paths*self.num_quests,1])

        outputs, path_state = tf.nn.dynamic_rnn(self.path_update,
                                                link_inputs,
                                                initial_state=self.flow_size,
                                                sequence_length=lens,
                                                dtype=tf.float32)
        self.ht = h_tild  # monitor
        self.li = link_inputs  # monitor
        self.outs=outputs # monitor

        path_state=tf.reshape(path_state,[self.num_quests,self.num_paths])


        return path_state
