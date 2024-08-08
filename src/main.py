# -*- coding: utf-8 -*-
from input import *
from model import *
import csv
def create_estimator():
    tf.logging.set_verbosity(tf.logging.INFO)
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    config = tf.estimator.RunConfig(
        save_summary_steps=100,
        save_checkpoints_steps=100,
        model_dir=MODEL_SAVE_PATH,
        keep_checkpoint_max=2,
        log_step_count_steps=100,
        session_config=session_config
    )
    model = CrossDQN()
    estimator = tf.estimator.Estimator(model_fn=model.model_fn_estimator, config=config)
    return estimator


def save_estimator(estimator, export_dir):
    def _serving_input_receiver_fn():
        receiver_tensors = {
            'user_id': tf.placeholder(tf.int64, [None, 1], name='user_id'),
            'behavior_poi_id_list': tf.placeholder(tf.int64, [None, 10], name='behavior_poi_id_list'),
            'ad_id_list': tf.placeholder(tf.int32, [None, 5], name='ad_id_list'),
            'oi_id_list': tf.placeholder(tf.int32, [None, 5], name='oi_id_list'),
            
            'context_id': tf.placeholder(tf.int32, [None, 1], name='context_id'),
            'action': tf.placeholder(tf.int32, [None, 5], name='action'),
            'ad_bid_id_list': tf.placeholder(tf.float32, [None, 5], name='ad_bid_id_list'),
            'oi_bid_id_list': tf.placeholder(tf.float32, [None, 5], name='oi_bid_id_list'),
            }
        return tf.estimator.export.ServingInputReceiver(receiver_tensors=receiver_tensors, features=receiver_tensors)
    export_dir = estimator.export_savedmodel(export_dir_base=export_dir, serving_input_receiver_fn=_serving_input_receiver_fn)
    return export_dir

def predict_with_model(estimator, input_fn):
    predictions = estimator.predict(input_fn=input_fn)
    csv_file_name = 'max_q_action_index.csv'
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write a header row (assuming self.max_q_action_index has a fixed size, e.g., 32)
        writer.writerow([f'index_{i}' for i in range(32)])  # Adjust the range according to the actual size

        for pred in predictions:
            max_q_action_index = pred['self.max_q_action_index']
            writer.writerow(max_q_action_index)

    print(f"max_q_action_index values saved to {csv_file_name}")
    for pred in predictions:
        print(pred)  # 打印每个预测结果

if __name__ == '__main__':
    estimator = create_estimator()
    train_input_fn = input_fn_maker(DATA_PATH)
    # 设定最大步数
    MAX_STEPS = 10
    # 训练模型
    estimator.train(input_fn=train_input_fn, steps=MAX_STEPS)
    save_estimator(estimator, PB_SAVE_PATH)

    # 定义预测输入函数
    predict_input_fn = input_fn_maker(DATA_PATH)  # 使用适当的预测数据

    # 获取并打印预测结果
    predict_with_model(estimator, predict_input_fn)
