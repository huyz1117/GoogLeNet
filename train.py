import os
import sys
import argparse
import time
import tensorflow as tf
from tensorflow.contrib import layers

from utils import load_image, parse_function, average_gradients, assign_to_device
from inceptionV1 import inception_v1


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="/home/skj/huyz/data/mini-imagenet/train", help="Path to the data directory")
    parser.add_argument("--num_epochs", type=int, default=500, help="The number of epochs to run")
    parser.add_argument("--batch_size", type=int, default=128, help="Number of images to process in a batch")
    parser.add_argument("--save_freq", type=int, default=15000, help="The frequence of saving checkpoint model")
    parser.add_argument("--lr_steps", type=list, default=[20000, 40000, 60000], help="Learning rate deacy steps")
    parser.add_argument("--image_height", type=int, default=224, help="The height of image in pixels")
    parser.add_argument("--image_width", type=int, default=224, help="The width of iamge in pixels")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="L2 weight regularization")
    parser.add_argument("--num_gpu", type=list, default=2, help="The GPU device to use")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory name to save training logs")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint", help="Directory name to save checkpoint model")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Initial learning rate")
    parser.add_argument("--summary_freq", type=int, default=300, help="The frequency of saving summary")
    parser.add_argument("--momentum", type=float, default=0.9, help="")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout rate")
    parser.add_argument("--gpu_device", default=[0, 1], help="GPU device to use")
    parser.add_argument("--max_steps", type=int, default=30000, help="The number of iterations")
    args = parser.parse_args()

    return args

all_classes, all_images, all_labels = load_image("/home/skj/huyz/data/mini-imagenet/train")
filenames = tf.constant(all_images)
labels = tf.constant(all_labels)
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(parse_function)
dataset = dataset.shuffle(len(all_images)).apply(tf.contrib.data.batch_and_drop_remainder(256)).repeat(500)
  
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()

def main(args):

    with tf.device("/cpu:0"):
        global_step = tf.Variable(0, trainable=False)
        images = tf.placeholder(tf.float32, shape=[None, args.image_height, args.image_width, 3], name="img_inputs")
        labels = tf.placeholder(tf.int64, shape=[None], name="img_labels")
        dropout_rate = tf.placeholder(tf.float32, name="dropout")

        # define learning rate schedule
        p = int(512.0/args.batch_size)
        # lr_steps = [p*val for val in args.lr_steps]
        lr_steps = [5000, 10000]
        lr = tf.train.piecewise_constant(global_step, boundaries=lr_steps, values=[0.0001, 0.00005, 0.00003], name='lr_schedule')
        # define optimize method
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        
        tower_grads = []
        loss_dict = {}
        loss_keys = []
        reuse_vars = False

        # Loop over all GPUs and construct their own computation graph
        for i in range(len(args.gpu_device)):
            with tf.device("/gpu:%d"%args.gpu_device[i]):
                # split data between GPUs
                _x = images[i * args.batch_size: (i+1) * args.batch_size]
                _y = labels[i * args.batch_size: (i+1) * args.batch_size]
                # print(_x.shape)
                # print(_y.shape)
                
                # Because Dropout have different behavior at training and prediction time, we
                # need to create 2 distinct computation graphs that share the same weights.
                
                # Create a graph for training
                ax1_out_train, ax2_out_train, main_out_train = inception_v1(_x, args.dropout, is_training=True, reuse=reuse_vars)
                
                # Create another graph for testing that reuse the same weights
                ax1_out_test, ax2_out_test, main_out_test = inception_v1(_x, args.dropout, is_training=False, reuse=True)
                
                
                # tf.nn.sparse_softmax_cross_entropy_with_logits()传入的logits为神经网络的输出，
                # shape为[batch_size, num_classes]
                # 传入的labels为一维向量，长度是batch_size，每一个值的取值为[0, num_classes)，
                # 每一个值代表对应样本的类别
                
                ax1_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=ax1_out_train, labels=_y))
                ax2_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=ax2_out_train, labels=_y))
                main_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=main_out_train, labels=_y))
                loss = main_loss + 0.3 * ax1_loss + 0.3 * ax2_loss
                
                loss_dict[('ax1_loss_%s_%d' % ('gpu', i))] = ax1_loss
                loss_keys.append(('ax1_loss_%s_%d' % ('gpu', i)))
                loss_dict[('ax2_loss_%s_%d' % ('gpu', i))] = ax2_loss
                loss_keys.append(('ax2_loss_%s_%d' % ('gpu', i)))
                loss_dict[('main_loss_%s_%d' % ('gpu', i))] = main_loss
                loss_keys.append(('main_loss_%s_%d' % ('gpu', i)))

                grads = optimizer.compute_gradients(loss)
                tower_grads.append(grads)

                # Only first GPU compute accuracy
                if i == 0:
                    pred = tf.nn.softmax(main_out_test)
                    # Evaluate model (with test logits, for dropout to be disabled)
                    correct_prediction = tf.equal(tf.argmax(pred, -1), _y)
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                
                reuse_vars = True
                    
        tower_grads = average_gradients(tower_grads)
        # Apply the gradients to adjust the shared variables.
        train_op = optimizer.apply_gradients(grads, global_step=global_step)
        
        # Start traininig
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # summary writer
            summary = tf.summary.FileWriter(args.log_dir, sess.graph)
            summaries = []
            # add grad histogram op
            for grad, var in grads:
                if grad is not None:
                    summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
            # add trainabel variable gradients
            for var in tf.trainable_variables():
                summaries.append(tf.summary.histogram(var.op.name, var))
            # add loss summary
            for keys, val in loss_dict.items():
                summaries.append(tf.summary.scalar(keys, val))
            # add learning rate
            summaries.append(tf.summary.scalar('leraning_rate', lr))
            # add train accuracy 
            summaries.append(tf.summary.scalar("train_acc", accuracy))
            summary_op = tf.summary.merge(summaries)
            
            init = tf.global_variables_initializer()
            sess.run(init)
            
            saver = tf.train.Saver()

            # restore checkpoint if it exists
            could_load = load_ckpt(sess, saver, args.checkpoint_dir)
            
            # begin iteration
            for step in range(1, args.max_steps+1):
                image_batch, label_batch = sess.run(one_element)
                feed_dict={images: image_batch, labels: label_batch}
                start = time.time()
                _, summary_op_val, train_loss, train_acc = sess.run([train_op, summary_op, loss, accuracy], feed_dict=feed_dict)
                end = time.time()
                print("Current step: %d/%d Time: %.4f Train_loss: %.6f Train_acc: %.4f"%(step, args.max_steps, end - start, train_loss, train_acc))

                if step % args.summary_freq == 0:
                    summary.add_summary(summary_op_val, step)
                
                # save the checkpoint per save_freq
                if step % args.save_freq == 0:
                    save_ckpt(sess, saver, args.checkpoint_dir, global_step)
# Reference: https://github.com/taki0112/Self-Attention-GAN-Tensorflow/blob/master/SAGAN.py
def load_ckpt(sess, saver, checkpoint_dir):
    import re
    print("[*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        print("[*] Success to read {}".format(ckpt_name))
        return True
    else:
        print("[*] Failed to find a checkpoint")
        return False

def save_ckpt(sess, saver, checkpoint_dir, step):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    saver.save(sess, os.path.join(checkpoint_dir, "inception.model"), global_step=step)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))

