import tensorflow as tf
import model
import input
import time
import numpy as np
import os
from datetime import datetime
from tensorflow.python.framework import graph_util

FLAGS = tf.app.flags.FLAGS


def train():
    x = tf.placeholder(tf.float32, shape=[None, FLAGS.height, FLAGS.width, FLAGS.depth], name='input_node')
    y = tf.placeholder(tf.float32, shape=[None, FLAGS.num_class], name="Y_label")
    is_training = tf.placeholder_with_default(False, shape=(), name='is_training')
    x_image = tf.summary.image('images', x, max_outputs=3)

    with tf.device('/cpu:0'):
        train_images, train_labels = input.get_data('train', FLAGS.data_dir, FLAGS.batch_size)
        valid_images, valid_labels = input.get_data('valid', FLAGS.data_dir, FLAGS.test_image_num)

    logits, y_pred = model.hypothesis(x, is_training)
    loss = model.cost(logits, y)

    with tf.variable_scope('optimizer') as scope:
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, name='optimizer').minimize(loss)

    accuracy = model.accuracy(y_pred, y)

    with tf.Session() as sess:

        saver = tf.train.Saver()
        merged_summary = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter(FLAGS.train_logs_dir)
        validate_writer = tf.summary.FileWriter(FLAGS.validate_logs_dir)
        train_writer.add_graph(sess.graph)

        # 모든 변수들을 초기화한다.
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        total_batch = int(FLAGS.train_image_num / FLAGS.batch_size)

        for epoch in range(FLAGS.epoch):
            t_avg_cost = 0
            t_avg_accuracy = 0

            v_avg_cost = 0
            v_avg_accuracy = 0

            for step in range(total_batch):
                t_images, t_labels = sess.run([train_images, train_labels])
                v_images, v_labels = sess.run([valid_images, valid_labels])

                start_time = time.time()

                _, t_loss, t_summary, t_accuracy = sess.run([optimizer, loss, merged_summary, accuracy],
                                                            feed_dict={x: t_images, y: t_labels, is_training: True})
                v_summary, v_accuracy, v_loss = sess.run([merged_summary, accuracy, loss],
                                                         feed_dict={x: v_images, y: v_labels, is_training: False})
                duration = time.time() - start_time

                assert not np.isnan(t_loss), 'Model diverged with loss = NaN'

                num_examples_per_step = FLAGS.batch_size + FLAGS.test_image_num
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d (%.1f examples/sec; %.3f ''sec/batch)')
                print(format_str % (datetime.now(), step+1, examples_per_sec, sec_per_batch))
                format_str = ('Train loss = %f, accuracy = %f')
                print(format_str % (t_loss, t_accuracy))
                format_str = ('Valid loss = %f, accuracy = %f')
                print(format_str % (v_loss, v_accuracy))

                t_avg_cost += t_loss / total_batch
                t_avg_accuracy += t_accuracy / total_batch

                v_avg_cost += v_loss / total_batch
                v_avg_accuracy += v_accuracy / total_batch

                # train_writer.add_summary(t_img, step)
                train_writer.add_summary(t_summary, (epoch * total_batch) + step)
                validate_writer.add_summary(v_summary, global_step=(epoch * total_batch) + step)

            print("************************************************************************")
            print("End Epoch : %04d, Train_set Cost = %f, Accuracy = %f" % (epoch + 1, t_avg_cost, t_avg_accuracy))
            print("End Epoch : %04d, Valid_set Cost = %f, Accuracy = %f" % (epoch + 1, v_avg_cost, v_avg_accuracy))
            print("************************************************************************")

            checkpoint_path = os.path.join(FLAGS.output_path, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=epoch)

        train_writer.close()
        validate_writer.close()

        coord.request_stop()
        coord.join(threads)

        # graph_def = tf.get_default_graph().as_graph_def()

        # Batch Norm Bug fix try...n n..
        graph_def = sess.graph.as_graph_def()
        for node in graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr: del node.attr['use_locking']

        output_graph = graph_util.convert_variables_to_constants(sess, graph_def, [FLAGS.output_node_name])
        with tf.gfile.GFile(os.path.join(FLAGS.output_path, 'cifar10_bn_gap_32' + '.pb'), 'wb') as f:
            f.write(output_graph.SerializeToString())

train()
