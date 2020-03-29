import hjson

class HyperParams():
    """
    Class that loads hyperparameters from a json file.
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """
        Saves parameters to json file
        """
        with open(json_path, 'w') as f:
            #json.dump(self.__dict__, f, indent=4)
            hjson.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """
        Loads parameters from json file
        """
        with open(json_path) as f:
            params = hjson.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """
        Gives dict-like access to HyperParams instance by
        `params.dict['learning_rate']`
        """
        return self.__dict__


import numpy as np
from sklearn.model_selection import train_test_split

class DataSet():

    def __init__(self, file_path, test_ratio):
        self.load(file_path, test_ratio)

    def load(self, file_path, test_ratio):
        with np.load(file_path) as f:
            x = f['x']
            y = f['y']
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio, random_state=0, shuffle=False)
            self.__dict__.update({'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test})

    @property
    def dict(self):
        return self.__dict__


import tensorflow as tf
import tensorflow.keras as keras

def cal_focal(y_true, y_pred, alpha=0.25, gamma=2, is_to_mask=True):
    if is_to_mask:
        y_true_m = tomask(y_true)
        y_pred_m = tomask(y_pred)
    else:
        y_true_m = y_true
        y_pred_m = y_pred
    y_true = tf.cast(y_true_m, tf.float32)
    y_pred = tf.cast(y_pred_m, tf.float32)
    sigmoid_p = tf.nn.sigmoid(y_pred)
    zeros = tf.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # y_true > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = tf.where(y_true > zeros, y_true - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # y_true > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = tf.where(y_true > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.math.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.math.log(
        tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)


class FocalLoss(keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2,
                 reduction=keras.losses.Reduction.AUTO,
                 name='focal_loss'):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        # with tf.name_scope("focal_loss"):
        # with tf.compat.v1.get_default_graph().as_default(), tf.name_scope("focal_loss"):  # it seems doens't work
        # y_true=tomask((y_true))
        # y_pred=tomask((y_pred))
        return cal_focal(y_true, y_pred, alpha=self.alpha, gamma=self.gamma, is_to_mask=False)


# @title Metrics { vertical-output: true, form-width: "40%", display-mode: "both" }
def flatten(x, dtype=None):
    x = tf.reshape(x, [-1])
    if dtype is not None:
        x = tf.cast(x, dtype)
    return x


def softargmax(x, beta=1e10):
    x = tf.cast(x, dtype=tf.float32)
    x_range = tf.range(x.shape[-1], dtype=tf.float32)
    res = tf.reduce_sum(tf.nn.softmax(x * beta) * x_range, axis=-1)
    res = tf.cast(res, dtype=tf.int32)
    return res


def tomask(input, issoft=False, axis=4):
    if issoft:
        # DEBUG: not working
        mask = softargmax(input)
    else:
        # mask = tf.argmax(input, axis=CHANNEL_AXIS, output_type=tf.int32)
        mask = tf.math.argmax(input, axis=axis, output_type=tf.int32)
    mask = tf.expand_dims(mask, axis)
    return mask


def cal_dice(y_true, y_pred, loss_type='jaccard', eps=1e-5, is_to_mask=True):
    # (Dice) Dice Coefficient
    if is_to_mask:
        y_true_m = tomask(y_true)
        y_pred_m = tomask(y_pred)
    else:
        y_true_m = y_true
        y_pred_m = y_pred
    # y_true=tf.convert_to_tensor(y_true)
    y_true_f = flatten(y_true_m, tf.float32)
    # y_pred=tf.convert_to_tensor(y_pred)
    y_pred_f = flatten(y_pred_m, tf.float32)
    intersection = tf.reduce_sum(y_pred_f * y_true_f)
    if loss_type == 'jaccard':
        union = tf.reduce_sum(tf.square(y_pred_f)) + tf.reduce_sum(tf.square(y_true_f))
    elif loss_type == 'sorensen':
        union = tf.reduce_sum(y_pred_f) + tf.reduce_sum(y_true_f)
    else:
        raise ValueError("Unknown `loss_type`: %s" % loss_type)
    dice = (2 * intersection + eps) / (union + eps)
    return dice


def cal_accuracy(y_true, y_pred):
    y_true_m = tomask(y_true)
    y_pred_m = tomask(y_pred)
    correct_prediction = tf.equal(y_true_m, y_pred_m)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def cal_voe(y_true, y_pred, eps=1e-5):
    # (VOE)  Volumetric Overlap Error
    y_true_f = flatten(tomask(y_true), tf.float32)
    y_pred_f = flatten(tomask(y_pred), tf.float32)
    intersection = tf.reduce_sum(y_pred_f * y_true_f)
    denominator = tf.reduce_sum(tf.clip_by_value(y_true_f + y_pred_f, 0, 1))
    voe = (1 - intersection / (denominator + eps))
    return tf.reduce_mean(voe)


def cal_rvd(y_true, y_pred, eps=1e-5):
    # (RVD)   Relative Volume Difference
    y_true_f = flatten(tomask(y_true), tf.float32)
    y_pred_f = flatten(tomask(y_pred), tf.float32)
    y_true_sum = tf.reduce_sum(y_true_f)
    y_pred_sum = tf.reduce_sum(y_pred_f)
    rvd = tf.abs(y_pred_sum - y_true_sum) / (y_true_sum + eps)
    return tf.reduce_mean(rvd)


class MyAccuracy(keras.metrics.Metric):
    def __init__(self, name='Accuracy', **kwargs):
        super(MyAccuracy, self).__init__(name=name, **kwargs)
        self.accuracy = self.add_weight(name=name, initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.accuracy.assign(cal_accuracy(y_true, y_pred))

    def result(self):
        return self.accuracy

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.accuracy.assign(0.)


class Dice(keras.metrics.Metric):
    def __init__(self, loss_type='jaccard', eps=1e-5, name='dice', **kwargs):
        super(Dice, self).__init__(name=name, **kwargs)
        self.dice = self.add_weight(name='dice', initializer='zeros')
        self.loss_type = loss_type
        self.eps = eps

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.dice.assign(cal_dice(y_true, y_pred, self.loss_type, self.eps))

    def result(self):
        return self.dice

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.dice.assign(0.)


class DiceLoss(keras.losses.Loss):
    def __init__(self, loss_type='jaccard', eps=1e-5,
                 reduction=keras.losses.Reduction.AUTO,
                 name='dice_loss'):
        super().__init__(reduction=reduction, name=name)
        self.loss_type = loss_type
        self.eps = eps

    def call(self, y_true, y_pred):
        dice = cal_dice(y_true, y_pred, self.loss_type, self.eps, is_to_mask=False)
        return (1 - dice)


class SoftCrossEntropyLoss(keras.losses.Loss):
    def __init__(self, reduction=keras.losses.Reduction.AUTO, name='SCE_loss'):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        # with tf.name_scope(self.name):
        # y_true = tomask(y_true)
        # y_pred = tomask(y_pred)
        sce = tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)
        return sce


from sklearn.metrics import roc_auc_score
class AUC(keras.metrics.Metric):
    def __init__(self, name='auc', **kwargs):
        super(AUC, self).__init__(name=name, **kwargs)
        self.auc = self.add_weight(name='auc', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        auc = tf.py_function(func=roc_auc_score, inp=[y_true, y_pred], Tout=tf.float32, name="sklearn/auc")
        self.auc.assign(auc)

    def result(self):
        return self.auc

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.auc.assign(0.)



import time
import matplotlib.pyplot as plt
class TBCallback(keras.callbacks.TensorBoard):
    def __init__(self,
                 log_dir='logs',
                 histogram_freq=1,
                 write_graph=True,
                 write_images=False,
                 update_freq='epoch',
                 profile_batch=2,
                 embeddings_freq=0,
                 embeddings_metadata=None,
                 **kwargs):
        super(TBCallback, self).__init__(log_dir,
                                         histogram_freq,
                                         write_graph,
                                         write_images,
                                         update_freq,
                                         profile_batch,
                                         embeddings_freq,
                                         embeddings_metadata,
                                         **kwargs)

    def on_epoch_begin(self, epoch, logs=None):
        super(TBCallback, self).on_epoch_begin(epoch, logs)
        self.epoch_time_start = time.time()
        return

    def on_epoch_end(self, epoch, logs=None):
        super(TBCallback, self).on_epoch_end(epoch, logs)
        period = time.time() - self.epoch_time_start
        print("\n\nTraing time {:7.4f} s".format(period))
        print("logs = ", logs)
        for i in self._writers:
            with self._get_writer(i).as_default():
                if i == "train":
                    lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                    tf.summary.scalar('Learning Rate', lr, step=epoch)

        # save_layer_pos=[0.3,0.4,0.5,0.6]

        # for i in self._writers:
        #    if i == "train":
        #        x = x_train
        #        y = y_train
        #    else:
        #        x = x_test
        #        y = y_test
        #    #DEBUG check: 1 has images, 0 might not have.
        #    x=x[1][tf.newaxis,...]
        #    y=y[1][tf.newaxis,...]
        #    result = self.model.predict(x)
        #    with self._get_writer(i).as_default():
        #        nlayers=result.shape[1]
        #        #plt.imsave("result.jpg",result[0,19,:,:,0])
        #        tf.summary.scalar('Dice',cal_dice(y,result),step=epoch)
        #        tf.summary.scalar('VOE',cal_voe(y,result),step=epoch)
        #        tf.summary.scalar('RVD',cal_rvd(y,result),step=epoch)

        #        #zero_m=tf.zeros_like(y)
        #        #print("zero dice for cal_dice is ", cal_dice(y, zero_m, loss_type="sorensen",eps=0, is_to_mask=False))
        #        #print("self dice for cal_dice is ", cal_dice(y, y, loss_type="sorensen",eps=0, is_to_mask=False))
        #        #print("zero dice for cal_dice2 is ", cal_dice2(y, zero_m, loss_type="sorensen",eps=0, is_to_mask=False))
        #        #print("self dice for cal_dice2 is ", cal_dice2(y, y, loss_type="sorensen",eps=0, is_to_mask=False))
        #        #print("zero dice with mask for cal_dice is ", cal_dice(y, zero_m, loss_type="sorensen",eps=0, is_to_mask=True))
        #        #print("self dice with mask for cal_dice is ", cal_dice(y, y, loss_type="sorensen",eps=0, is_to_mask=True))
        #        #print("zero dice with mask for cal_dice2 is ", cal_dice2(y, zero_m, loss_type="sorensen",eps=0, is_to_mask=True))
        #        #print("self dice with mask for cal_dice2 is ", cal_dice2(y, y, loss_type="sorensen",eps=0, is_to_mask=True))

        #        maskdice=1-cal_dice(y, result, loss_type="jaccard",eps=0, is_to_mask=True)
        #        if maskdice>0.95:
        #            plt.imsave("y_true_mask.jpg", tomask(y)[0,19,:,:,0])
        #            plt.imsave("y_pred_mask.jpg", tomask(result)[0,19,:,:,0])
        #        print(i, "dice loss for cal_dice is ", 1-cal_dice(y, result, loss_type="jaccard",eps=0, is_to_mask=False))
        #        print(i, "dice loss with mask for cal_dice is ", 1-cal_dice(y, result, loss_type="jaccard",eps=0, is_to_mask=True))

        #        true_mask = tomask(y)
        #        pred_mask = tomask(result)
        #        print("{} true mask max {}".format(i, tf.reduce_max(true_mask)))
        #        print("{} pred mask max {}".format(i, tf.reduce_max(pred_mask)))
        #        for k in save_layer_pos:
        #            si=int(nlayers * k)
        #            tf.summary.image(i + "/Prediction", self.extract_layer_image(layer=pred_mask, batch_i=0, slice_i=si, feature_i=0), step=epoch)
        #            tf.summary.image(i + "/Truth", self.extract_layer_image(layer=true_mask, batch_i=0, slice_i=si, feature_i=0), step=epoch)

        # with self._get_writer(i).as_default():
        #    #for i in layer_outputs:
        #    #    print(i.name)
        #    #layers=[]
        #    #for i in ['Output/Conv3D', 'Output']:
        #    #    layers.append(self.model.get_layer(i+"/Identity:0").output)
        #    #print("y test max is ", tf.reduce_max(y_test))
        #    #pred_mask = tf.argmax(y_test, axis=CHANNEL_AXIS, output_type=tf.int32)
        #    #pred_mask = pred_mask[..., tf.newaxis]
        #    #print("y test pred mask shape is ", pred_mask.shape)
        #    #for k in [0.3,0.4,0.5,0.6]:

        #    for j,value in enumerate(result):
        #        nlayers=value.shape[1]
        #        if layer_outputs[j].name=="Output/Identity:0":
        #            pred_mask = tomask(value)
        #            for k in [0.3, 0.4, 0.5, 0.6]:
        #                #tf.compat.v2.summary.image(scope + "/" + layer_outputs[i].name,
        #                #                       self.extract_layer_image(layer=value, batch_i=0, slice_i=int(nlayers * k),
        #                #                                                feature_i=0), step=epoch)
        #                #tf.compat.v2.summary.image(scope + "/" + layer_outputs[i].name+"/1", self.extract_layer_image(layer=value, batch_i=0, slice_i=int(nlayers * k),
        #                #                                    feature_i=1), step=epoch)
        #                #res=np.asarray(res)
        #                #pred_mask=tf.argmax(value, axis=CHANNEL_AXIS, output_type=tf.int32)
        #                tf.summary.image(i + "/" + "Prediction",
        #                                 self.extract_layer_image(layer=pred_mask, batch_i=0, slice_i=int(nlayers * k),
        #                                                          feature_i=0), step=epoch)

        #            if i == "train":
        #                y=y_train
        #            else:
        #                y=y_test
        #            true_mask = tomask(y)
        #            tf.summary.scalar('Dice',tf.reduce_mean(cal_dice(true_mask,pred_mask)),step=epoch)
        #            tf.summary.scalar('VOE',tf.reduce_mean(cal_voe(true_mask,pred_mask)),step=epoch)
        #            tf.summary.scalar('RVD',tf.reduce_mean(cal_rvd(true_mask,pred_mask)),step=epoch)
        #            for k in [0.3,0.4,0.5,0.6]:
        #                tf.summary.image(i + "/" + "Truth", self.extract_layer_image(layer=true_mask, batch_i=1, slice_i=int(nlayers * k), feature_i=0), step=epoch)

    def mask_to_image(self, mask):
        my_cm = plt.cm.get_cmap('gray')
        m_max = np.max(mask)
        m_min = np.min(mask)
        if abs(m_max) < 1e-5:
            norm_mask = mask
        else:
            norm_mask = (mask - m_min) / (m_max - m_min)
        img = my_cm(norm_mask)
        return img

    def extract_layer_image(self, layer, batch_i=0, slice_i=0, feature_i=0):
        '''
        extract one layer image from CNN 3d network

        :param layer: which network layer
        :param feature_i: which feature of the network layer
        :param img_layer: image layer
        :return: 2d image
        '''

        img = layer[batch_i, slice_i, :, :, feature_i]
        img = self.mask_to_image(img)
        img = tf.expand_dims(img, 0)
        img_name = "batch{}_slice{}_feature{}".format(batch_i, slice_i, feature_i)
        return tf.reshape(img, img.shape, img_name)
