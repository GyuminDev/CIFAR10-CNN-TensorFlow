import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('raw_height', 32, '')
tf.app.flags.DEFINE_integer('raw_width', 32, '')
tf.app.flags.DEFINE_integer('height', 32, '')
tf.app.flags.DEFINE_integer('width', 32, '')
tf.app.flags.DEFINE_integer('depth', 3, '')
tf.app.flags.DEFINE_integer('num_class', 10, '')
tf.app.flags.DEFINE_integer('conv1_filter_num', 32, '')
tf.app.flags.DEFINE_integer('conv1_filter_size', 3, '')
tf.app.flags.DEFINE_integer('conv2_filter_num', 64, '')
tf.app.flags.DEFINE_integer('conv2_filter_size', 3, '')
tf.app.flags.DEFINE_integer('conv3_filter_num', 128, '')
tf.app.flags.DEFINE_integer('conv3_filter_size', 3, '')
tf.app.flags.DEFINE_integer('conv4_filter_num', 256, '')
tf.app.flags.DEFINE_integer('conv4_filter_size', 3, '')
tf.app.flags.DEFINE_integer('conv5_filter_num', 512, '')
tf.app.flags.DEFINE_integer('conv5_filter_size', 3, '')
tf.app.flags.DEFINE_integer('epoch', 9, '')
tf.app.flags.DEFINE_integer('batch_size', 512, '')
tf.app.flags.DEFINE_integer('train_image_num', 50000, '')
tf.app.flags.DEFINE_integer('test_image_num', 10000, '')
tf.app.flags.DEFINE_float('learning_rate', 0.001, '')
tf.app.flags.DEFINE_boolean('use_fp16', False, '')
tf.app.flags.DEFINE_integer('eval_interval_secs', 1,'')

# FloydHub Config
tf.app.flags.DEFINE_string('data_dir', '/data', '')
tf.app.flags.DEFINE_string('checkpoint_dir', '/output', '')
tf.app.flags.DEFINE_string('input_node_name', 'input_node', '')
tf.app.flags.DEFINE_string('output_node_name', 'output', '')
tf.app.flags.DEFINE_string('output_path', '/output', '')
tf.app.flags.DEFINE_string('train_logs_dir', '/output/train', '')
tf.app.flags.DEFINE_string('validate_logs_dir', '/output/valid', '')

# Local Config
# tf.app.flags.DEFINE_string('data_dir', os.getcwd() + '/data', '')
# tf.app.flags.DEFINE_string('checkpoint_dir', os.getcwd() + '/output', '')
# tf.app.flags.DEFINE_string('input_node_name', 'input_node', '')
# tf.app.flags.DEFINE_string('output_node_name', 'output', '')
# tf.app.flags.DEFINE_string('output_path', os.getcwd() + '/output', '')
# tf.app.flags.DEFINE_string('train_logs_dir', os.getcwd() + '/output/train', '')
# tf.app.flags.DEFINE_string('validate_logs_dir', os.getcwd() + '/output/valid', '')