import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def _variable_on_cpu(name, shape, initializer):
    """cpu 메모리에 변수를 선언하고 return.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """
    tf.truncated_normal_initializer()를 이용해서 weight(filter포함)을 초기화 하여 return
    L2 Regularization을 진행한다. weight이 클 수록 패널티를 부과하여 오버피팅을 억제하는 방법

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        # tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
        tf.contrib.layers.xavier_initializer())
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var


def BatchNorm(inputT, is_training=True, scope=None):
    # Note: is_training is tf.placeholder(tf.bool) type
    return tf.cond(is_training,
                   lambda: tf.contrib.layers.batch_norm(inputT, is_training=True, center=True,
                                                        updates_collections=None, scope=scope,
                                                        decay=0.9, zero_debias_moving_mean=True, reuse=None),
                   lambda: tf.contrib.layers.batch_norm(inputT, is_training=False, center=True,
                                                        updates_collections=None, scope=scope,
                                                        decay=0.9, zero_debias_moving_mean=True, reuse=True))


def hypothesis(images, is_training):
    '''
    Conv1 - relu - pool - norm
    Conv2 - relu - norm - pool
    Conv3 - relu - norm
    Conv4 - relu - norm - pool
    2개의 F.C Layer - hypothesis

    :param images: input image matrix, shape = [batch_size, width, height, 3]
    :return: hypothesis 결과, shape = [batch_size, class_num]
    '''

    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[FLAGS.conv1_filter_size, FLAGS.conv1_filter_size,
                                                    FLAGS.depth, FLAGS.conv1_filter_num],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        batch_norm1 = BatchNorm(inputT=conv, is_training=is_training, scope='batch_norm')
        conv1 = tf.nn.relu(batch_norm1, name=scope.name)

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[FLAGS.conv2_filter_size, FLAGS.conv2_filter_size,
                                                    FLAGS.conv1_filter_num, FLAGS.conv2_filter_num],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        batch_norm2 = BatchNorm(inputT=conv, is_training=is_training, scope='batch_norm')
        conv2 = tf.nn.relu(batch_norm2, name=scope.name)

    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # conv3
    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[FLAGS.conv3_filter_size, FLAGS.conv3_filter_size,
                                                    FLAGS.conv2_filter_num, FLAGS.conv3_filter_num],
                                             stddev=5e-2,
                                             wd=0.0)

        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        batch_norm3 = BatchNorm(inputT=conv, is_training=is_training, scope='batch_norm')
        conv3 = tf.nn.relu(batch_norm3, name=scope.name)

    pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    # conv4
    with tf.variable_scope('conv4') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[FLAGS.conv4_filter_size, FLAGS.conv4_filter_size,
                                                    FLAGS.conv3_filter_num, FLAGS.conv4_filter_num],
                                             stddev=5e-2,
                                             wd=0.0)

        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        batch_norm4 = BatchNorm(inputT=conv, is_training=is_training, scope='batch_norm')
        conv4 = tf.nn.relu(batch_norm4, name=scope.name)

    pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool4')  # 1, 8, 8, 256

    with tf.variable_scope('conv5') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[FLAGS.conv5_filter_size, FLAGS.conv5_filter_size,
                                                    FLAGS.conv4_filter_num, FLAGS.conv5_filter_num],
                                             stddev=5e-2,
                                             wd=0.0)

        conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
        batch_norm5 = BatchNorm(inputT=conv, is_training=is_training, scope='batch_norm')
        conv5 = tf.nn.relu(batch_norm5, name=scope.name)

    # global average pooling
    with tf.variable_scope('global_average_pooling') as scope:
        filter_size = conv5.get_shape()[1].value
        gap = tf.nn.avg_pool(conv5, ksize=[1, filter_size, filter_size, 1],
                             strides=[1, filter_size, filter_size, 1], padding='VALID')

    # local3
    with tf.variable_scope('local1') as scope:
        # dim = reshape.get_shape()[1].value
        dim = 1
        for d in gap.get_shape()[1:].as_list():
            dim *= d

        reshape = tf.reshape(gap, [-1, dim])
        weights = _variable_with_weight_decay('weights', shape=[dim, FLAGS.num_class],
                                              stddev=0.04, wd=0.0001)
        biases = _variable_on_cpu('biases', [FLAGS.num_class], tf.constant_initializer(0.1))
        hypothesis = tf.add(tf.matmul(reshape, weights), biases, name="logits")

    output = tf.nn.softmax(hypothesis, name=FLAGS.output_node_name)

    tf.summary.histogram('hypothesis', hypothesis)
    tf.summary.histogram('output', output)

    return hypothesis, output


def cost(hypothesis, labels):
    '''
    :param hypothesis: model을 통해 예측된 가설
    :param labels: 실제 Y Label
    :return: croos-entropy(오차<cost>)
    '''
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=hypothesis, labels=labels, name='loss'))
        tf.summary.scalar("loss", cross_entropy)
        tf.add_to_collection('loss', cross_entropy)

    return cross_entropy


def accuracy(output, labels):
    '''
    :param hypothesis: 가설함수
    :param labels: 실제 Y Label
    :return: accuracy(정확도)
    '''
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(output, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('accuracy', accuracy)
    return accuracy
