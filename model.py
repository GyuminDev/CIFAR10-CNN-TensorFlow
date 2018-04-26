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
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var

def hypothesis(images):
    '''
    Conv1 - relu - pool - norm
    Conv2 - relu - norm - pool
    Conv3 - relu - norm
    Conv4 - relu - norm - pool
    2개의 F.C Layer - hypothesis
    norm은 tf.nn.lrn()을 사용하여 local response normalization을 진행한다.
    형성된 Filter의 수를 사용해서 filter 간에 정규화를 의미하는 것,
    :param images: input image matrix, shape = [batch_size, width, height, 3]
    :return: hypothesis 결과, shape = [batch_size, class_num]
    '''

    with tf.variable_scope('conv1') as scope:
        # 3x3 크기, 3개의 색상(RGB)의 필터를 64개 선언.
        kernel = _variable_with_weight_decay('weights',
                                             shape=[FLAGS.conv1_filter_size, FLAGS.conv1_filter_size,
                                                    FLAGS.depth, FLAGS.conv1_filter_num],
                                             stddev=5e-2,
                                             wd=0.0)
        # input image shape = [batch_size, width, height, depth]
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [FLAGS.conv1_filter_num], tf.constant_initializer(0.0))
        conv1 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)

        # variable_summaries(conv1)
        w1_hist = tf.summary.histogram("conv1_W", kernel)
        b1_hist = tf.summary.histogram("conv1_biases", biases)

    # pool1
    # padding 옵션 SAME
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    # norm1
    # local_response_normalization 진행(64개의 filter를 정규화하는 의미)
    norm1 = tf.nn.local_response_normalization(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')
    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[FLAGS.conv2_filter_size, FLAGS.conv2_filter_size,
                                                    FLAGS.conv1_filter_num, FLAGS.conv2_filter_num],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [FLAGS.conv2_filter_num], tf.constant_initializer(0.1))
        conv2 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)

        # variable_summaries(conv2)
        w2_hist = tf.summary.histogram("conv2_W", kernel)
        b2_hist = tf.summary.histogram("conv2_biases", biases)

    # norm2
    norm2 = tf.nn.local_response_normalization(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # conv3
    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[FLAGS.conv3_filter_size, FLAGS.conv3_filter_size,
                                                    FLAGS.conv2_filter_num, FLAGS.conv3_filter_num],
                                             stddev=5e-2,
                                             wd=0.0)

        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [FLAGS.conv3_filter_num], tf.constant_initializer(0.1))
        conv3 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)

        # variable_summaries(conv2)
        w3_hist = tf.summary.histogram("conv3_W", kernel)
        b3_hist = tf.summary.histogram("conv3_biases", biases)

    # norm3
    norm3 = tf.nn.local_response_normalization(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm3')
    # pool3
    # pool3 shape = [128, 32, 32, 64]
    pool3 = tf.nn.max_pool(norm3, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    # conv4
    with tf.variable_scope('conv4') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[FLAGS.conv4_filter_size, FLAGS.conv4_filter_size,
                                                    FLAGS.conv3_filter_num, FLAGS.conv4_filter_num],
                                             stddev=5e-2,
                                             wd=0.0)
        # conv shape = [128, 64, 64, 64]
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [FLAGS.conv4_filter_num], tf.constant_initializer(0.1))
        conv4 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)

        # variable_summaries(conv2)
        w4_hist = tf.summary.histogram("conv3_W", kernel)
        b4_hist = tf.summary.histogram("conv3_biases", biases)

    # norm3
    norm4 = tf.nn.lrn(conv4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm3')
    # pool3
    pool4 = tf.nn.max_pool(norm4, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    # local3
    with tf.variable_scope('local1') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        # reshape = tf.reshape(pool2, [FLAGS.batch_size, -1]) # reshape = [128, 6*6*64]
        # dim = reshape.get_shape()[1].value # 2304
        dim = 1
        for d in pool4.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(pool4, [-1, dim])
        weights = _variable_with_weight_decay('weights', shape=[dim, 512],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
        local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

        # variable_summaries(local3)
        w5_hist = tf.summary.histogram("local3_W", weights)
        b5_hist = tf.summary.histogram("local3_biases", biases)

    with tf.variable_scope('hypothesis') as scope:
        weights = _variable_with_weight_decay('weights', [512, FLAGS.num_class], stddev=1 / 192.0, wd=0.0)
        biases = _variable_on_cpu('biases', [FLAGS.num_class], tf.constant_initializer(0.0))
        hypothesis = tf.add(tf.matmul(local1, weights), biases, name="logits")

        w6_hist = tf.summary.histogram("W", weights)
        b6_hist = tf.summary.histogram("biases", biases)
        tf.summary.histogram('hypothesis', hypothesis)

    output = tf.nn.softmax(hypothesis, name=FLAGS.output_node_name)

    return hypothesis, output


def cost(hypothesis, labels):
    '''
    :param hypothesis: model을 통해 예측된 가설
    :param labels: 실제 Y Label
    :return: croos-entropy(오차<cost>)
    '''
    with tf.variable_scope('loss') as scope:
        # labels = tf.cast(labels, tf.int32)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=hypothesis, labels=labels, name='loss'))
        tf.summary.scalar("loss", cross_entropy)
        tf.add_to_collection('loss', cross_entropy)

    return cross_entropy


def optimizer(cross_entropy):
    '''
    :param cross_entropy: cross-Entropy를 이용해 계산된 오차<cost>
    :return: AdamOptimizer를 이용하여 train을 진행
    '''
    with tf.variable_scope('optimizer') as scope:
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, name='optimizer').minimize(cross_entropy)

    return optimizer


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
