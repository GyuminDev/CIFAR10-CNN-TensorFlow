import tensorflow as tf
import config
import os
import tarfile
import sys
from six.moves import urllib

FLAGS = tf.app.flags.FLAGS
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1200
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 60


def read_raw_images(filenames):
    '''
    :param filenames: .bin 파일 이름 목록
    :return: uint8image: 8-bit unsigned integer, (0~255) 0, 1을 1bit로 표현 가능하며, 총 8개(8bit)로 표현 2^8
             label: label
    '''
    # filename = ['./data/' + data_set + '_data.bin']
    # filename Queue에 filenames에 들어있는 파일 이름을 넣음
    filename_queue = tf.train.string_input_producer(filenames, shuffle=True)

    image_bytes = FLAGS.raw_height * FLAGS.raw_width * FLAGS.depth  # image_bytes = input height x input width x 3(RGB)
    record_bytes = image_bytes + 1  # record_bytes = width*height*3+(label size)

    # reader 객체로 filename queue에 접근하여 읽어온 data를 value에 넣음.
    # value에 들어있는 data를 unit8 형식으로 decode
    # decode한 dats는 queue에 쌓이게 된다.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)

    # tf.slice(input_, begin, size, name=None)
    # label = input의 begin부터 size만큼 자른다. 첫 번째 1바이트는 label이므로 int로 cast해서 생성
    # http://blog.naver.com/PostView.nhn?blogId=wjddudwo209&logNo=220976695912&categoryNo=75&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=search
    label = tf.cast(tf.slice(record_bytes, [0], [1]), tf.int64)
    # 1번 index(label을 제외한 나머지)부터 image_bytes 만큼 자른 후 [height x width x 3] -> [3, width, height]로 reshape
    depth_major = tf.reshape(tf.slice(record_bytes, [1], [image_bytes]),
                             [FLAGS.depth, FLAGS.raw_width, FLAGS.raw_height])
    # tf.transpose()
    # [0, 1, 2]를 [1, 2, 0]으로 바꾼다고 이해하면 됨.
    uint8image = tf.transpose(depth_major, [1, 2, 0])

    return uint8image, label


def generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    '''
    tf.train.shuffle_batch()는 queue내부에서 shuffle을 진행한다.
    :param image: 3-D Tensor, [width, height, 3]
    :param label: 1-D Tensor
    :param min_queue_examples:
    :param batch_size: int 128
    :param shuffle: boolean
    :return:
            image : 4-D Tensor [batch_size, width, height, 3]
            tf.reshape(label_batch, [batch_size]) : 1-D Tensor [batch_size]
    '''

    # shuffle하기 위한 thread 갯수 선언
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)

    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    return images, tf.one_hot(tf.reshape(label_batch, [batch_size]), FLAGS.num_class)


def distorted_inputs(data_dir, batch_size):
    '''
    read_raw_images()를 이용해 읽어온 Tensor를 이용하여
    128 x 128로 변경 및 왜곡시키고 표준화한 train data set image를 생성
    :param data_dir: input Binary File Path
    :param batch_size: batchSize
    :return:
    '''

    # os.path.join()을 이용해서 경로를 붙여줌.
    # bin파일들의 이름을 저장
    # filenames = [os.path.join(data_dir, 'data.bin')]

    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 5)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # image Tensor를 float형으로 형변환
    image, label = read_raw_images(filenames)
    reshaped_image = tf.cast(image, tf.float32)
    # label = tf.cast(label, tf.float32)

    # random하게 자른 후 좌우 반전, 밝기 조절, 대비 조절을 랜덤하게 적용 후
    # 이렇게 왜곡을 주어서 데이터 세트의 크기를 키운다. 그리고 화이트닝을 적용한다.(모델이 둔감해 지도록)
    # Overfitting을 방지
    # distorted_image = tf.random_crop(reshaped_image, [FLAGS.height, FLAGS.width, FLAGS.depth])
    distorted_image = tf.image.random_flip_left_right(reshaped_image)
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    float_image = tf.image.per_image_standardization(distorted_image)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(FLAGS.train_image_num *
                             min_fraction_of_examples_in_queue)

    return generate_image_and_label_batch(float_image, label, min_queue_examples, batch_size, shuffle=True)


def inputs(data_dir, batch_size):
    '''
    test image set을 불러 올 때 호출 됨.
    이미지를 왜곡하지 않으며 shuffle 하지 않음(수정 가능)
    :param data_dir:
    :param batch_size:
    :return:
    '''
    filenames = [os.path.join(data_dir, 'test_batch.bin')]
    # filenames = [os.path.join(data_dir, 'test_batch.bin')]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    image, label = read_raw_images(filenames)
    reshaped_image = tf.cast(image, tf.float32)

    # resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, FLAGS.height, FLAGS.width)
    float_image = tf.image.per_image_standardization(reshaped_image)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(FLAGS.test_image_num *
                             min_fraction_of_examples_in_queue)

    return generate_image_and_label_batch(float_image, label, min_queue_examples, batch_size, shuffle=True)


def get_data(data_set, data_dir, batch_size):
    if data_set is 'train':
        return distorted_inputs(data_dir, batch_size)
    else:
        return inputs(data_dir, batch_size)