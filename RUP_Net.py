import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tensorflow.keras as keras
import util
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import pandas as pd

class RUP_Net(object):
    def __init__(self, hyperParams):
        print("tensorflow version: ", tf.__version__)
        self.hyperParams=hyperParams
        self.dataset = util.DataSet(hyperParams['INPUT_FILE'], hyperParams['TEST_RATIO']).dict

        self.train_set = tf.data.Dataset.from_tensor_slices((self.dataset['x_train'], self.dataset['y_train']))
        self.test_set = tf.data.Dataset.from_tensor_slices((self.dataset['x_test'], self.dataset['y_test']))
        self.augment(hyperParams['AUGMENT_EPOCH'])
        self.model = self.build_net()
        if hyperParams['LOSS'] == "focal":
            loss = util.FocalLoss(alpha=0.25, gamma=2, name="focal_loss"),
        elif hyperParams['LOSS'] == "dice":
            loss = util.DiceLoss(loss_type='jaccard', eps=1e-5, name='dice_loss')
        else:
            loss = keras.losses.binary_crossentropy(from_logits=True)

        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.hyperParams['INIT_LEARNING_RATE']),
                      loss=loss,
                      metrics=['binary_accuracy', util.MyAccuracy(name="accuracy"), util.Dice(loss_type='jaccard', eps=1e-5, name='dice')])
        ## model.summary()
        keras.utils.plot_model(self.model, 'RUP-Net.png', show_shapes=True)

        self.log_dir = self.hyperParams['LOG_DIR'] + "/" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


    def train(self):
        outputFolder = os.path.dirname(self.hyperParams['CKPT_FILE'])
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)

        SHUFFLE_SIZE = self.hyperParams['BATCH_SIZE'] * np.prod(self.dataset['y_train'].shape)
        self.train_set = self.train_set.batch(self.hyperParams['BATCH_SIZE']).shuffle(SHUFFLE_SIZE)
        print(self.train_set.element_spec)

        #csvlog = keras.callbacks.CSVLogger("train.csv",separator=',', append=False)

        tb_callback = TBCallback(log_dir=self.log_dir, write_images=False,histogram_freq=0, profile_batch=0, update_freq=self.hyperParams['SAVE_FREQ'], dataset=self.dataset)
        checkpointer = keras.callbacks.ModelCheckpoint(filepath=self.hyperParams['CKPT_FILE'],
                                                      monitor='val_dice',
                                                      mode='max',
                                                      verbose=1,
                                                      save_best_only=True,
                                                      save_weights_only=True
                                                      )
        dynamic_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1)
        cb = [tb_callback, checkpointer, dynamic_lr]
        if self.hyperParams['EARLY_STOP']:
            earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            cb.append(earlystop)
        history = self.model.fit(self.train_set, epochs=self.hyperParams['EPOCH'], validation_data=(self.dataset['x_test'], self.dataset['y_test']),
                                 validation_steps=self.hyperParams['BATCH_SIZE'],
                                 callbacks=cb)

        zero_m = tf.zeros_like(self.dataset['y_test'])
        print("zero accuracy for cal_accuracy is ", util.cal_accuracy(self.dataset['y_test'], zero_m))
        print("self accuracy for cal_accuracy is ", util.cal_accuracy(self.dataset['y_test'], self.dataset['y_test']))
        print("zero dice for cal_dice is ", util.cal_dice(self.dataset['y_test'], zero_m, is_to_mask=False, loss_type='jaccard'))
        print("self dice for cal_dice is ", util.cal_dice(self.dataset['y_test'], self.dataset['y_test'], is_to_mask=False, loss_type='jaccard'))
        print("zero focal for cal_focal is ", util.cal_focal(self.dataset['y_test'], zero_m, is_to_mask=False))
        print("self focal for cal_focal is ", util.cal_focal(self.dataset['y_test'], self.dataset['y_test'], is_to_mask=False))
        print("zero focal mask for cal_focal is ", util.cal_focal(self.dataset['y_test'], zero_m, is_to_mask=True))
        print("self focal mask for cal_focal is ", util.cal_focal(self.dataset['y_test'], self.dataset['y_test'], is_to_mask=True))
        print("history is ", history)

        pred = self.predict(self.hyperParams['IS_PRINT_INTERMEDIATE_LAYER'])

        print("accuracy for pred is ", util.cal_accuracy(self.dataset['y_test'], pred))
        print("dice for pred is ", util.cal_dice(self.dataset['y_test'], pred, is_to_mask=True, loss_type='jaccard'))
        print("mask dice for pred is ", util.cal_dice(self.dataset['y_test'], pred, is_to_mask=False, loss_type='jaccard'))

        outputFolder = os.path.dirname(self.hyperParams['CSV_FILE'])
        if not os.path.exists(outputFolder):
           os.makedirs(outputFolder)
        with open(self.hyperParams['CSV_FILE'], mode='w') as f:
           hist_df = pd.DataFrame(history.history)
           hist_df.to_csv(f, index_label="epoch")

    def predict(self, isPrintIntermediateLayer=True):
        self.model.load_weights(self.hyperParams['CKPT_FILE'])
        pred = self.model.predict(self.dataset['x_test'])

        if isPrintIntermediateLayer:
            layer_names=[]
            layer_outputs=[]
            for layer in self.model.layers:
                if 'Conv' in layer.name:
                    layer_names.append(layer.name)
                    layer_outputs.append(layer.output)
            inter_model = keras.models.Model(inputs=self.model.input, outputs=layer_outputs)
            inter_pred = inter_model.predict(self.dataset['x_test'])
            inter_file_writer = tf.summary.create_file_writer(self.log_dir+"/prediction")
            with inter_file_writer.as_default():
                for name, layer_image in zip(layer_names, inter_pred):
                    #layer_image = util.tomask(layer_image)
                    tf.summary.image("prediction/"+name, util.extract_layer_image(layer=layer_image, batch_i=0, slice_i=int(layer_image.shape[1]*0.5), feature_i=0), step=0)

        return pred

    def u_net(self, input_layer, filters, conv_size, name=None, isres=True, ispool=False,
              layer_depth=4, deconv="deconv", isadd=True):
        x = input_layer
        downlist = []  # including input/output of each node in down-sampling operation

        # down-sampling
        for i in range(layer_depth - 1):
            node_name = "/Down_" + str(i)
            x = self.node(x, filters * (2 ** i), conv_size, name + node_name, isres, isadd, filter_mul=1)
            downlist.append(x)
            if ispool:
                x = keras.layers.MaxPooling3D(padding="same", name=name + node_name + "/Pool")(x)

        # bottom
        node_name = "/Bottom"
        # x=node(x, filters*(2**(layer_depth-1)), conv_size, name+node_name, isres, isadd)
        x = self.node(x, filters * (2 ** (layer_depth - 2)), conv_size, name + node_name, isres, isadd)

        def decoder(x, filters, conv_size, strides, name, deconv="deconv"):
            decname = "/DeConv_Conv3DTrans"
            if deconv == "deconv":
                x = keras.layers.Conv3DTranspose(filters, conv_size, strides=strides, padding="same",
                                                 kernel_initializer='he_normal',
                                                 bias_initializer='he_uniform',
                                                 # kernel_regularizer=keras.regularizers.l2(0.01),
                                                 # bias_regularizer=keras.regularizers.l2(0.01),
                                                 name=name + decname)(x)
            elif deconv == "upsampling":
                decname = "/DeConv_UpSampling"
                # this 2 is same as one used for MaxPooling, so if pool changes, this will change
                x = keras.layers.UpSampling3D(strides, name=name + decname)(x)
            else:
                # implenment PDN here
                x = x
            # x = keras.layers.BatchNormalization(name=name+decname+"/BN")(x)
            # x = keras.layers.Activation('relu',name=name+decname+"/Activation")(x)
            return x

        # up-sampling
        strides = int(ispool) + 1
        for i in range(layer_depth - 2, -1, -1):  # backward like 2,1,0 if dep=4
            node_name = "/Up_" + str(i)
            x = decoder(x, filters * (2 ** i), conv_size, strides, name + node_name, deconv)
            x, cname = self.comb_layer(layers=[x, downlist[i]], isadd=isadd, scope=name + node_name + "/Bridge")
            # no pooling in up-sampling
            x = self.node(x, filters * (2 ** i), conv_size, name + node_name, isres, isadd=isadd, filter_mul=0.5)

        return x

    def comb_layer(self, layers, isadd=True, scope=""):
        if isadd:
            addname = "/Add"
            x = keras.layers.Add(name=scope + addname)(layers)
        else:
            addname = "/Concat"
            x = keras.layers.Concatenate(name=scope + addname)(layers)
        return x, scope + addname

    def node(self, input_layer, filters, conv_size, name=None, isres=True, isadd=True, filter_mul=1):
        # filter_mul: multiple filter by filter_mul**i
        # with tf.name_scope(name): #it seems doens't work
        x = input_layer
        num_sublayers = 2
        for i in range(num_sublayers):
            str_i = str(i)
            # x = keras.layers.Conv3D((num_sublayers-i)*filters, conv_size, activation='relu', padding='same',
            #                        kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01),
            #                        name=name+"/Conv3D_"+str_i)(x)
            x = keras.layers.Conv3D(int(filters * (filter_mul ** i)), conv_size, activation=None, padding='same',
                                    kernel_initializer='he_normal',
                                    bias_initializer='he_uniform',
                                    # kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01),
                                    name=name + "/Conv3D_" + str_i)(x)
            # x = keras.layers.BatchNormalization(name=name+"/BN_"+str_i)(x)
            # x = keras.layers.Activation("relu",name=name+"/Activation_"+str_i)(x)
            # x = keras.layers.Dropout(0.5, name=name+"/Dropout_"+str_i)(x)

        if isres:
            y = input_layer
            if isadd:
                y_channel = y.shape[-1]
                x_channel = x.shape[-1]
                mul = y_channel / x_channel
                if mul < 1:
                    mul = 1 / mul
                mul = int(mul)
                if mul != 1:
                    y = tf.tile(input=y, multiples=tf.constant([1, 1, 1, 1, mul]))
            x, cname = self.comb_layer(layers=[x, y], isadd=isadd, scope=name)
            # x = keras.layers.BatchNormalization(name=cname+"/Comb_BN")(x)

        return x

    def build_net(self):
        #DEBUG: x_train
        input_layer = keras.Input(shape=self.dataset['x_test'].shape[1:], name="Input")
        x = input_layer
        x = keras.layers.BatchNormalization(name="Input/BN")(x)

        # #use Conv3D to make (?, 64,64,64,1) to (?,64,64,64,32) #x = keras.layers.Conv3D(START_NUM_OF_FILTERS,
        # kernel_size=3, padding="same",activation='relu', name="Input/Conv3D")(x) #use Concat to make (?, 64,64,64,
        # 1) to (?,64,64,64,32) for i in range(START_NUM_OF_FILTERS-1): x=keras.layers.Concatenate(
        # axis=CHANNEL_AXIS)([x, input_layer]) x = u_net(input_layer=x, filters=START_NUM_OF_FILTERS, conv_size=3,
        # name = "U-Net", isres=ISRES, ispool=ISPOOL, layer_depth=UNET_DEPTH, deconv=DECONV, isadd=ISADD)

        # y=x
        # for i in range(2):
        #  tmp = u_net(input_layer=y, filters=START_NUM_OF_FILTERS, conv_size=3+2*i, name = "U{}-Net".format(i), isres=ISRES, ispool=ISPOOL,
        #            layer_depth=UNET_DEPTH, deconv=DECONV, isadd=ISADD)
        #  if ISADD:
        #    x=keras.layers.Add(name="U{}Add".format(i))([x,tmp])
        #  else:
        #    x=keras.layers.Concatenate(name="U{}Contac".format(i))([x,tmp])

        for i in range(1):
            x = self.u_net(input_layer=x, filters=self.hyperParams['START_NUM_OF_FILTERS'], conv_size=3, name="U{}-Net".format(i),
                           isres=self.hyperParams['ISRES'], ispool=self.hyperParams['ISPOOL'], layer_depth=self.hyperParams['UNET_DEPTH'],
                           deconv=self.hyperParams['DECONV'], isadd=self.hyperParams['ISADD'])

        # for i in range(3):
        #  x = u_net(input_layer=x, filters=START_NUM_OF_FILTERS, conv_size=3+i*2, name = "U{}-Net".format(i), isres=ISRES, ispool=ISPOOL,
        #            layer_depth=UNET_DEPTH, deconv=DECONV, isadd=ISADD)
        #  if ISADD:
        #    x=keras.layers.Add(name="U{}Add".format(i))([x,input_layer])
        #  else:
        #    x=keras.layers.Concatenate(name="U{}Contac".format(i))([x,input_layer])

        x = keras.layers.Conv3D(self.hyperParams['NUM_CLASS'], kernel_size=1, padding="same", name="Output/Conv3D")(x)
        x = keras.layers.BatchNormalization(name="Output")(x)

        model = keras.Model(input_layer, x, name="RUP-Net")
        return model

    def plot_dataset(self, dataset, n_cases=8, n_slices_per_case=10, start_case=1):
        num_of_sets = len(dataset.element_spec)
        img_shape = dataset.element_spec[0].shape
        # [0]=slice/depth, [1]=width, [2]=height, [3]=color
        d, w, h, c = 0, 1, 2, 3
        dep, wid, hei = img_shape[d], img_shape[w], img_shape[h]
        slices = np.rint(np.linspace(0, 1, n_slices_per_case) * (dep - 1)).astype(np.int32)
        output = np.zeros((num_of_sets, hei * n_slices_per_case, wid * n_cases))

        i = 0
        dataset = dataset.skip(start_case - 1)
        for case in dataset.take(n_cases):
            for j in range(num_of_sets):
                input = case[j].numpy()
                input = input[slices, :, :, 0]
                output[j, :, i * wid:(i + 1) * wid] = np.vstack(input)
            i += 1

        fig, ax = plt.subplots(1, num_of_sets, figsize=(15, 15))
        for i in range(num_of_sets):
            # plt.imshow(output, extent=[1,n_cases, slices[-1], slices[0]])
            img = ax[i].imshow(output[i, :, :], cmap="gray", aspect="auto")
            xtick = np.arange(1, n_cases + 1)
            ytick = slices / (slices[1] - slices[0])
            ax[i].set_xticks(xtick * wid - wid * 0.5)
            ax[i].set_xticklabels(xtick)
            ax[i].set_yticks(ytick * hei + hei * 0.5)
            ax[i].set_yticklabels(slices)
            # ax[i].set_title("Title")
            ax[i].set_xlabel("Case #")
            ax[i].set_ylabel("Slice #")
            fig.show(img)


    # @title
    def aug_rotate(self, img):
        nb = img.shape[0]
        outimg = []
        for i in range(nb):
            outimg.append(tf.image.rot90(img[i, :, :, :], tf.random.uniform([], minval=1, maxval=4, dtype=tf.int32)))
        outimg = tf.stack(outimg)
        return outimg


    def aug_flip(self, img):
        nb = img.shape[0]
        outimg = []
        for i in range(nb):
            outimg.append(tf.image.flip_left_right(img[i, :, :, :]))
        outimg = tf.stack(outimg)
        return outimg


    def dataset_len(self, dataset):
        i = 0
        for e in dataset:
            i += 1
        return i


    def augment(self, augment_epoch):
        if augment_epoch > 0:
            augmentations = [self.aug_flip, self.aug_rotate]

            print("Before augmentation, train_set size = ", self.dataset_len(self.train_set))
            orig = self.train_set
            for i in range(augment_epoch):
                for f in augmentations:
                    self.train_set = self.train_set.concatenate(
                        orig.map(lambda x, y: (f(x), f(y)), num_parallel_calls=tf.data.experimental.AUTOTUNE))
            print("After augmentation, train_set size = ", self.dataset_len(self.train_set))



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
                 dataset=None,
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
        self.dataset=dataset

    def on_epoch_begin(self, epoch, logs=None):
        super(TBCallback, self).on_epoch_begin(epoch, logs)
        self.epoch_time_start = time.time()
        return

    def on_epoch_end(self, epoch, logs=None):
        super(TBCallback, self).on_epoch_end(epoch, logs)
        period = time.time() - self.epoch_time_start
        print("\n\nTraing time {:7.4f} s".format(period))
        print("logs = ", logs)
        if epoch % self.update_freq == 0:
            for i in self._writers:
                with self._get_writer(i).as_default():
                    if i == "train":
                        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                        tf.summary.scalar('Learning Rate', lr, step=epoch)

            for i in self._writers:
                if i == "train":
                    x = self.dataset['x_train']
                    y = self.dataset['y_train']
                    summary_slice = self.dataset['train_slice']
                else:
                    x = self.dataset['x_test']
                    y = self.dataset['y_test']
                    summary_slice = self.dataset['test_slice']
                #DEBUG check: 1 has images, 0 might not have.
                x=x[0][tf.newaxis,...]
                y=y[0][tf.newaxis,...]
                result = self.model.predict(x)
                with self._get_writer(i).as_default():
                    nlayers=result.shape[1]
                    #plt.imsave("result.jpg",result[0,19,:,:,0])
                    tf.summary.scalar('Dice',util.cal_dice(y,result),step=epoch)
                    tf.summary.scalar('VOE',util.cal_voe(y,result),step=epoch)
                    tf.summary.scalar('RVD',util.cal_rvd(y,result),step=epoch)

                #    #zero_m=tf.zeros_like(y)
                #    #print("zero dice for cal_dice is ", cal_dice(y, zero_m, loss_type="sorensen",eps=0, is_to_mask=False))
                #    #print("self dice for cal_dice is ", cal_dice(y, y, loss_type="sorensen",eps=0, is_to_mask=False))
                #    #print("zero dice for cal_dice2 is ", cal_dice2(y, zero_m, loss_type="sorensen",eps=0, is_to_mask=False))
                #    #print("self dice for cal_dice2 is ", cal_dice2(y, y, loss_type="sorensen",eps=0, is_to_mask=False))
                #    #print("zero dice with mask for cal_dice is ", cal_dice(y, zero_m, loss_type="sorensen",eps=0, is_to_mask=True))
                #    #print("self dice with mask for cal_dice is ", cal_dice(y, y, loss_type="sorensen",eps=0, is_to_mask=True))
                #    #print("zero dice with mask for cal_dice2 is ", cal_dice2(y, zero_m, loss_type="sorensen",eps=0, is_to_mask=True))
                #    #print("self dice with mask for cal_dice2 is ", cal_dice2(y, y, loss_type="sorensen",eps=0, is_to_mask=True))

                    #maskdice=1-cal_dice(y, result, loss_type="jaccard",eps=0, is_to_mask=True)
                    #if maskdice>0.95:
                    #plt.imsave("y_true_mask.jpg", util.tomask(y)[0,19,:,:,0])
                    #plt.imsave("y_pred_mask.jpg", util.tomask(result)[0,19,:,:,0])
                    if i == "train":
                        plt.imsave("y_true_mask.jpg", y[0,summary_slice,:,:,0])
                        plt.imsave("y_pred_mask.jpg", result[0,summary_slice,:,:,0])
                        print("result shape is ",result.shape)
                    #print(i, "dice loss for cal_dice is ", 1-cal_dice(y, result, loss_type="jaccard",eps=0, is_to_mask=False))
                    #print(i, "dice loss with mask for cal_dice is ", 1-cal_dice(y, result, loss_type="jaccard",eps=0, is_to_mask=True))

                    #print("{} true max {}".format(i, tf.reduce_max(y)))
                    #print("{} pred max {}".format(i, tf.reduce_max(result)))
                    true_mask = util.tomask(y)
                    pred_mask = util.tomask(result)
                    #print("{} true mask max {}".format(i, tf.reduce_max(true_mask)))
                    #print("{} pred mask max {}".format(i, tf.reduce_max(pred_mask)))

                    tf.summary.image(i + "/Prediction", util.extract_layer_image(layer=pred_mask, batch_i=0, slice_i=summary_slice, feature_i=0), step=epoch)
                    tf.summary.image(i + "/Truth", util.extract_layer_image(layer=true_mask, batch_i=0, slice_i=summary_slice, feature_i=0), step=epoch)
