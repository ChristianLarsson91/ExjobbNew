import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import pc_util
import nn
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnetFinal_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=128, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--visu', action='store_true', help='Whether to dump image for error case [default: False]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = 4
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))]

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/kittiTrain.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/kittiTest.txt'))



def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate(num_votes):
    is_training = False

    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
        loss = MODEL.get_loss(pred, labels_pl, end_points)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss}

    eval_one_epoch(sess, ops, num_votes)


def eval_one_epoch(sess, ops, num_votes=1, topk=1):
    error_cnt = 0
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    fout = open(os.path.join(DUMP_DIR, 'pred_label.txt'), 'w')

    current_data, order_list = nn.exportData()
    current_label = np.zeros(len(current_data))
    current_data = current_data[:,0:NUM_POINT,:]
    current_label = np.squeeze(current_label)
    print(current_data.shape)
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    print(file_size)
    predValues = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        cur_batch_size = end_idx - start_idx

        # Aggregating BEG
        batch_loss_sum = 0 # sum of losses for the batch
        batch_pred_sum = np.zeros((cur_batch_size, NUM_CLASSES)) # score for classes
        batch_pred_classes = np.zeros((cur_batch_size, NUM_CLASSES)) # 0/1 for classes
        for vote_idx in range(num_votes):
            rotated_data = provider.rotate_point_cloud_by_angle(current_data[start_idx:end_idx, :, :],
                                              vote_idx/float(num_votes) * np.pi * 2)
            feed_dict = {ops['pointclouds_pl']: rotated_data,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}
            loss_val, pred_val = sess.run([ops['loss'], ops['pred']],
                                      feed_dict=feed_dict)
            batch_pred_sum += pred_val
            batch_pred_val = np.argmax(pred_val, 1)
            for el_idx in range(cur_batch_size):
                batch_pred_classes[el_idx, batch_pred_val[el_idx]] += 1
            batch_loss_sum += (loss_val * cur_batch_size / float(num_votes))
        pred_val_topk = np.argsort(batch_pred_sum, axis=-1)[:,-1*np.array(range(topk))-1]
        pred_val = np.argmax(batch_pred_classes, 1)
        pred_val = np.argmax(batch_pred_sum, 1)
        predValues.append(np.ndarray.tolist(pred_val)[0])
        # Aggregating END

        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        # correct = np.sum(pred_val_topk[:,0:topk] == label_val)
        total_correct += correct
        total_seen += cur_batch_size
        loss_sum += batch_loss_sum
        for i in range(start_idx, end_idx):
            l = int(current_label[i])
        
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i-start_idx] == l)
            fout.write('%d, %d\n' % (pred_val[i-start_idx], l))

            if pred_val[i-start_idx] != l and FLAGS.visu: # ERROR CASE, DUMP!
                img_filename = '%d_label_%s_pred_%s.jpg' % (error_cnt, SHAPE_NAMES[l],
                                                       SHAPE_NAMES[pred_val[i-start_idx]])
                img_filename = os.path.join(DUMP_DIR, img_filename)
                output_img = pc_util.point_cloud_three_views(np.squeeze(current_data[i, :, :]))
                scipy.misc.imsave(img_filename, output_img)
                error_cnt += 1

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    print(predValues)
    class_accuracies = np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)
    for i, name in enumerate(SHAPE_NAMES):
        log_string('%10s:\t%0.3f' % (name, class_accuracies[i]))

#canvas = scene.SceneCanvas(keys='interactive', show=True)
#view = canvas.central_widget.add_view()
#fov = 60.
#cam1 = scene.cameras.FlyCamera(parent=view.scene, fov=fov)
#cam2 = scene.cameras.TurntableCamera(parent=view.scene, fov=fov)
#cam3 = scene.cameras.ArcballCamera(parent=view.scene, fov=fov)
#view.camera = cam1

# Implement key presses
#@canvas.events.key_press.connect
#def on_key_press(event):
#    global opaque_cmap, translucent_cmap
#    if event.text == '1':
#        cam_toggle = {cam1: cam2, cam2: cam3, cam3: cam1}
#        view.camera = cam_toggle.get(view.camera, 'fly')
#    elif event.text == '2':
#        methods = ['mip', 'translucent', 'iso', 'additive']
#        method = methods[(methods.index(volume1.method) + 1) % 4]
#        print("Volume render method: %s" % method)
#        cmap = opaque_cmap if method in ['mip', 'iso'] else translucent_cmap
#        volume1.method = method
#        volume1.cmap = cmap
#        volume2.method = method
#        volume2.cmap = cmap
#    elif event.text == '3':
#        volume1.visible = not volume1.visible
#        volume2.visible = not volume1.visible
#    elif event.text == '4':
#        if volume1.method in ['mip', 'iso']:
#            cmap = opaque_cmap = next(opaque_cmaps)
#        else:
#            cmap = translucent_cmap = next(translucent_cmaps)
#        volume1.cmap = cmap
#        volume2.cmap = cmap
#    elif event.text == '0':
#        cam1.set_range()
#        cam3.set_range()
#    elif event.text != '' and event.text in '[]':
#        s = -0.025 if event.text == '[' else 0.025
#        volume1.threshold += s
#        volume2.threshold += s
#        th = volume1.threshold if volume1.visible else volume2.threshold
#        print("Isosurface threshold: %0.3f" % th)


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=1) 
#    data,labels = retrieveData()    
#    array = convertToNumpy2D(data)
#    scatter = scene.visuals.Markers()
#    scatter.set_data(array[:,:3], face_color=colorMaping(predValues,data),size=1)
#    view.add(scatter)
#    view.camera = scene.PanZoomCamera(aspect=1)
#    view.camera.set_range()
#    app.run()
    LOG_FOUT.close()

