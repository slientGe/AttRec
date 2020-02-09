import sys
import os
sys.path.append('..')
os.environ["CUDA_VISIBLE_DEVICES"]='1'

import tensorflow as tf
import numpy as np
from model_AttRec import AttRec
from make_datasets import make_datasets
from DataInput import DataIterator


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('file_path', 'input/u.data', 'training data dir')
tf.app.flags.DEFINE_string('test_path', 'input/test.csv', 'testing data dir')
tf.app.flags.DEFINE_string('train_path', 'input/train.csv', 'training data dir')
tf.app.flags.DEFINE_string('mode', 'train', 'train or test')
tf.app.flags.DEFINE_float('w', 0.3, 'The final score is a weighted sum of them with the controlling factor ω')
tf.app.flags.DEFINE_integer('num_epochs', 30, 'number of epochs')
tf.app.flags.DEFINE_integer('sequence_length', 5, 'sequence length')
tf.app.flags.DEFINE_integer('target_length', 3, 'target length')
tf.app.flags.DEFINE_integer('neg_sample_count',10, 'number of negative sample')
tf.app.flags.DEFINE_integer('item_count', 1685, 'number of items')
tf.app.flags.DEFINE_integer('user_count', 945, 'number of user')
tf.app.flags.DEFINE_integer('embedding_size', 100, 'embedding size')
tf.app.flags.DEFINE_integer('batch_size', 256, 'batch size')
tf.app.flags.DEFINE_float('learning_rate', 1e-2, 'learning rate')
tf.app.flags.DEFINE_float('keep_prob', 0.5, 'keep prob of dropout')
tf.app.flags.DEFINE_float('l2_lambda', 1e-3, 'Regularization rate for l2')
tf.app.flags.DEFINE_float('gamma', 0.5, 'gamma of the margin higle loss')
tf.app.flags.DEFINE_float('grad_clip', 10, 'gradient clip to prevent from grdient to large')
tf.app.flags.DEFINE_string('save_path','save_path/model1.ckpt','the whole path to save the model')

def Metric_HR(target_list, predict_list, num):
    count = 0
    for i in range(len(target_list)):
        t = target_list[i]
        preds = predict_list[i]
        preds = preds[:num]
        if t in preds:
            count += 1
    return count / len(target_list)

def Metric_MRR(target_list, predict_list):

    count = 0
    for i in range(len(target_list)):
        t = target_list[i]
        preds = predict_list[i]
        rank = preds.index(t) + 1
        count += 1 / rank
    return count / len(target_list)


def main(args):

    print(' make datasets')

    train_data, test_data ,user_all_items, all_user_count\
        , all_item_count, user_map, item_map \
        = make_datasets(FLAGS.file_path, FLAGS.target_length, FLAGS.sequence_length, isSave=False)
    FLAGS.item_count = all_item_count
    FLAGS.user_count = all_user_count
    all_index = [i for i in range(FLAGS.item_count)]

    print(' load model and training')
    with tf.Session() as sess:

        #Load model
        model = AttRec(FLAGS)
        topk_index = model.predict(all_index,len(all_index))
        total_loss = model.loss

        #Add L2
        # with tf.name_scope('l2loss'):
        #     loss = model.loss
        #     tv = tf.trainable_variables()
        #     regularization_cost = FLAGS.l2_lambda * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])
        #     total_loss = loss + regularization_cost

        #Optimizer
        global_step = tf.Variable(0, trainable=False)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(total_loss, tvars), FLAGS.grad_clip)
            grads_and_vars = tuple(zip(grads, tvars))
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        #Saver and initializer
        saver = tf.train.Saver()
        if FLAGS.mode == 'test':
            saver.restore(sess, FLAGS.save_path)
        else:
            sess.run(tf.global_variables_initializer())

        #Batch reader
        trainIterator = DataIterator(data=train_data
                                     , batch_size=FLAGS.batch_size
                                     ,max_seq_length=FLAGS.batch_size
                                     ,neg_count=FLAGS.neg_sample_count
                                     ,all_items=all_index
                                     ,user_all_items=user_all_items
                                     ,shuffle=True)
        testIterator = DataIterator(data=test_data
                                     ,batch_size = FLAGS.batch_size
                                     , max_seq_length=FLAGS.batch_size
                                     , neg_count=FLAGS.neg_sample_count
                                     , all_items=all_index
                                     , user_all_items=user_all_items
                                     , shuffle=False)
        #Training and test for every epoch
        for epoch in range(FLAGS.num_epochs):
            cost_list = []
            for train_input in trainIterator:
                user, next_target, user_seq, sl, neg_seq = train_input
                feed_dict = {model.u_p: user, model.next_p: next_target, model.sl: sl,
                             model.hist_seq: user_seq, model.neg_p: neg_seq,
                             model.keep_prob:FLAGS.keep_prob,model.is_Training:True}

                _, step, cost = sess.run([train_op, global_step, total_loss], feed_dict)
                cost_list.append(np.mean(cost))
            mean_cost = np.mean(cost_list)
            saver.save(sess, FLAGS.save_path)

            pred_list = []
            next_list = []
            # test and cal hr50 and mrr
            for test_input in testIterator:
                user, next_target, user_seq, sl, neg_seq = test_input
                feed_dict = {model.u_p: user, model.next_p: next_target, model.sl: sl,
                             model.hist_seq: user_seq,model.keep_prob:1.0
                            ,model.is_Training:False}
                pred_indexs = sess.run(topk_index, feed_dict)
                pred_list += pred_indexs.tolist()
                #only predict one next item
                single_target = [item[0] for item in next_target]
                next_list += single_target
            hr50 = Metric_HR(next_list,pred_list,50)
            mrr = Metric_MRR(next_list,pred_list)
            print(" epoch {},  mean_loss{:g}, test HR@50: {:g}, test MRR: {:g}"
                  .format(epoch + 1, mean_cost,hr50,mrr))



if __name__ == '__main__':
    tf.app.run()


