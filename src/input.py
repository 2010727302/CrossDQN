# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from config import *
import csv

def generate_random_demo_data():
    demo_data = []
    ad_bid_data = []
    oi_vol_data = []
    ad_vol_data = []
    for i in range(10):
        # 随机生成数据
        user_id = np.random.randint(100, 10001)
        history_poi_id_list = np.random.randint(100, 10001, size=10).tolist()
        ad_id_list = np.random.randint(10, 101, size=5).tolist()
        oi_id_list = np.random.randint(10, 101, size=5).tolist()
        ad_bid_id_list = np.random.uniform(10, 101, size=5).tolist()  # 随机生成大于0的bid值
        oi_bid_id_list = [0.0] * 5  # 生成全为0的bid值
        context_id = np.random.randint(1, 101)
        history_poi_id_list[4:] = [0] * (len(history_poi_id_list) - 4)
        oi_id_list[1]=321
        # 轮流使用最后五个数
        if i % 2 == 0:
            last_five_values = [1, 2, 3, 51, 2]
        else:
            last_five_values = [1, 2, 3, 32, 2]

        R_ad, R_fee, R_ex = 1.2, 2.0, 2.1  # 固定的R_ad, R_fee, R_ex

        # 拼接数据
        feature_data = [
            user_id, *history_poi_id_list, *ad_id_list, *oi_id_list, *ad_bid_id_list, *oi_bid_id_list,context_id,
            *last_five_values
        ]
        label_data = [R_ad, R_fee, R_ex]
        ad_bid_data.append(ad_bid_id_list)
        ad_vol_data.append(ad_id_list)
        oi_vol_data.append(oi_id_list)
        demo_data.append([feature_data, label_data])
    csv_file_name = 'ad_bid_id_list.csv'
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ad_bid_1", "ad_bid_2", "ad_bid_3", "ad_bid_4", "ad_bid_5"])
        writer.writerows(ad_bid_data)
    print(f"ad_bid_id_list saved to {csv_file_name}")
    csv_file_name = 'ad_id_list.csv'
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ad_id_1", "ad_id_2", "ad_id_3", "ad_id_4", "ad_id_5"])
        writer.writerows(ad_vol_data)
    print(f"ad_id_list saved to {csv_file_name}")
    csv_file_name = 'oi_id_list.csv'
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["oi_id_1", "oi_id_2", "oi_id_3", "oi_id_4", "oi_id_5"])
        writer.writerows(oi_vol_data)
    print(f"oi_id_list saved to {csv_file_name}")

    return demo_data
def demo_tfrecord_data(tfrecord_file_name):
    # user_id, history_poi_id_list, ad_id_list, oi_id_list, context_id, R_ad, R_fee, R_ex
    # demo_data = [
    #     [[121, 123, 456, 789, 321, 0, 0, 0, 0, 0, 0, 23, 31, 42, 22, 67, 76, 321, 36, 93, 22, 13, 1, 2, 3, 51, 2],
    #      [1.2, 2, 2.1]],
    #     [[121, 123, 456, 789, 321, 0, 0, 0, 0, 0, 0, 23, 31, 42, 22, 67, 76, 321, 36, 93, 22, 13, 1, 2, 3, 32, 2],
    #      [1.2, 2, 2.1]]
    # ]
    demo_data = generate_random_demo_data()
    print(demo_data)
    writer = tf.io.TFRecordWriter(tfrecord_file_name)
    for item in demo_data:
        feature_data = item[0]
        label_data = item[1]

        # 分割 feature_data 为 int64 和 float 部分
        feature_int64 = feature_data[:21] + feature_data[-6:]  # 假设前21个和最后6个是 int64 类型
        feature_float = feature_data[21:31]

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'feature_int64': tf.train.Feature(int64_list=tf.train.Int64List(value=feature_int64)),
                    'feature_float': tf.train.Feature(float_list=tf.train.FloatList(value=feature_float)),
                    'label': tf.train.Feature(float_list=tf.train.FloatList(value=label_data))
                }
            )
        )
        writer.write(example.SerializeToString())
    
    writer.close()
    print("finish to write data to tfrecord file!")

def generate_parse_tfrecord_local_fn():
    def _parse_function(batch_examples):
        data_description = {
            "feature_int64": tf.FixedLenFeature([27], dtype=tf.int64),  # 修改为实际的 int64 特征长度
            "feature_float": tf.FixedLenFeature([10], dtype=tf.float32),  # 修改为实际的 float 特征长度
            "label": tf.FixedLenFeature([3], dtype=tf.float32)
        }
        parsed_features = tf.parse_example(
            batch_examples,
            features=data_description
        )
        
        feature_int64_buffer = parsed_features['feature_int64']
        feature_float_buffer = parsed_features['feature_float']

        features = {
            'user_id': tf.cast(tf.gather(feature_int64_buffer, list(range(0, 1)), axis=1), tf.int64),
            'behavior_poi_id_list': tf.cast(tf.gather(feature_int64_buffer, list(range(1, 11)), axis=1), tf.int64),
            'ad_id_list': tf.cast(tf.gather(feature_int64_buffer, list(range(11, 16)), axis=1), tf.int64),
            'oi_id_list': tf.cast(tf.gather(feature_int64_buffer, list(range(16, 21)), axis=1), tf.int64),
            'context_id': tf.cast(tf.gather(feature_int64_buffer, list(range(21, 22)), axis=1), tf.int64),
            'action': tf.cast(tf.gather(feature_int64_buffer, list(range(22, 27)), axis=1), tf.int32),
            'ad_bid_id_list': tf.cast(tf.gather(feature_float_buffer, list(range(0, 5)), axis=1), tf.float32),
            'oi_bid_id_list': tf.cast(tf.gather(feature_float_buffer, list(range(5, 10)), axis=1), tf.float32)
        }
        
        label_buffer = parsed_features['label']
        labels = {
            'r_ad': tf.gather(label_buffer, list(range(1)), axis=1),
            'r_fee': tf.gather(label_buffer, list(range(1, 2)), axis=1),
            'r_ex': tf.gather(label_buffer, list(range(2, 3)), axis=1)
        }
        return features, labels

    return _parse_function



def input_fn_maker(file_names):
    def input_fn():
        _parse_fn = generate_parse_tfrecord_local_fn()
        files = tf.data.Dataset.list_files(file_names)
        dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=4 * 10))
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(buffer_size=BATCH_SIZE * 10)
        dataset = dataset.repeat(EPOCH)
        dataset = dataset.map(_parse_fn, num_parallel_calls=NUM_PARALLEL)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    return input_fn



if __name__ == '__main__':
    # print(DATA_PATH[0])
    demo_tfrecord_data(DATA_PATH[0])
    train_input_fn = input_fn_maker(DATA_PATH)
    features, labels = train_input_fn()
    sess = tf.Session()
    try:
        features_np, labels_np = sess.run([features, labels])
        print("*" * 100, "features_np")
        for key in features_np:
            print("=" * 50, key, np.shape(features_np[key]))
            print(features_np[key])
        print("*" * 100, "labels_np")
        for key in labels_np:
            print("=" * 50, key, np.shape(labels_np[key]))
            print(labels_np[key])
    except Exception as e:
        print(e)
