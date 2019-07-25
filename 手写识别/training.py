# coding:utf-8
# Mute tensorflow debugging information console
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.layers import Conv2D, MaxPooling2D, Convolution2D, Dropout, Dense, Flatten, LSTM
from keras.models import Sequential, save_model
from keras.utils import np_utils
from scipy.io import loadmat
import pickle
import argparse
import keras
import numpy as np

def load_data(mat_file_path, width=28, height=28, max_=None, verbose=True):
    ''' Load data in from .mat file as specified by the paper.

        Arguments:
            mat_file_path: path to the .mat, should be in sample/

        Optional Arguments:
            width: specified width
            height: specified height
            max_: the max number of samples to load
            verbose: enable verbose printing

        Returns:
            A tuple of training and test data, and the mapping for class code to ascii value,
            in the following format:
                - ((training_images, training_labels), (testing_images, testing_labels), mapping)

    '''
    # Local functions
    def rotate(img):
        # Used to rotate images (for some reason they are transposed on read-in) 旋转图像
        flipped = np.fliplr(img)
        return np.rot90(flipped)

    def display(img, threshold=0.5):
        # Debugging only
        render = ''
        str = ''
        for row in img:
            for col in row:
                if col > threshold:
                    render += '@'
                else:
                    render += '.'
            render += '\n'
        return render

    # Load convoluted list structure form loadmat
    mat = loadmat(mat_file_path)

    # Load char mapping
    mapping = {kv[0]:kv[1:][0] for kv in mat['dataset'][0][0][2]}
    pickle.dump(mapping, open('bin/mapping.p', 'wb' ))

    # 加载训练数据样本
    if max_ == None:
        max_ = len(mat['dataset'][0][0][0][0][0][0])  # 求出数据的数据集中图像个数
    training_images = mat['dataset'][0][0][0][0][0][0][:max_].reshape(max_, height, width, 1)  # 原始图像
    training_labels = mat['dataset'][0][0][0][0][0][1][:max_]           # 数据的标签

    print(training_images.shape)
    # np.savetxt('f.txt',training_images[23], fmt="%3d", delimiter="")
    # print(display(training_images[0]),training_labels[0])
    # print(display(training_images[23]),training_labels[23])
    # print(training_images[0],training_labels[0])

    # 加载测试数据样本
    if max_ == None:
        max_ = len(mat['dataset'][0][0][1][0][0][0])
    else:
        max_ = int(max_ / 6)
    testing_images = mat['dataset'][0][0][1][0][0][0][:max_].reshape(max_, height, width, 1)
    testing_labels = mat['dataset'][0][0][1][0][0][1][:max_]

    # Reshape training data to be valid
    if verbose == True: _len = len(training_images)
    for i in range(len(training_images)):
        if verbose == True: print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1)/_len) * 100), end='\r')
        training_images[i] = rotate(training_images[i])
    if verbose == True: print('')

    # Reshape testing data to be valid
    if verbose == True: _len = len(testing_images)
    for i in range(len(testing_images)):
        if verbose == True: print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1)/_len) * 100), end='\r')
        testing_images[i] = rotate(testing_images[i])
    if verbose == True: print('')

    # Convert type to float32
    training_images = training_images.astype('float32')
    testing_images = testing_images.astype('float32')

    # Normalize to prevent issues with model
    training_images /= 255
    testing_images /= 255

    nb_classes = len(mapping)

    return ((training_images, training_labels), (testing_images, testing_labels), mapping, nb_classes)

def build_net(training_data, width=28, height=28, verbose=False):
    ''' Build and train neural network. Also offloads the net in .yaml and the
        weights in .h5 to the bin/.

        Arguments:
            training_data: the packed tuple from load_data()

        Optional Arguments:
            width: specified width
            height: specified height
            epochs: the number of epochs to train over
            verbose: enable verbose printing
    '''
    # Initialize data
    (x_train, y_train), (x_test, y_test), mapping, nb_classes = training_data
    print(mapping)
    print(nb_classes)
    input_shape = (height, width, 1)  # 黑白通道

    # 超参数
    nb_filters = 32     # 使用的卷积滤波器的数量
    pool_size = (2, 2)  # 最大池化的池区大小
    kernel_size = (3, 3)    # 过滤器（卷积核）大小

    # 构建顺序模型
    model = Sequential()
    '''
    模型需要知道输入数据的shape,
    因此，Sequential的第一层需接受一个输入数据的参数，
    后面各层可以自动推导出中间数据的shape,
    不需要为第个层都指定这个参数
    '''
    # 第一层卷积层， 构造32个过滤器，第个过滤器覆盖范围是 3*3*1
    # 过滤器挪动步长为1，图像四周补一圈0，并用relu进行非线性变换
    model.add(Convolution2D(nb_filters,
                            kernel_size,
                            strides=(1, 1),
                            padding='valid',
                            input_shape=input_shape,
                            activation='relu'))
    print(model.output)

    model.add(Convolution2D(nb_filters,
                            kernel_size,
                            activation='relu'))
    print(model.output)
    # 添加一层 MaxPooling，在2*2的格子中取最大值
    model.add(MaxPooling2D(pool_size=pool_size))
    print(model.output)
    # 设立Dropout层，将Dropout的概率设为0.25
    model.add(Dropout(0.25)) # 神经单元随机失活概率
    print(model.output)

    # 把当前层节点展平，拉成一维数据，才能进行全连接操作
    model.add(Flatten())
    print(model.output)
    # 全连接层1
    model.add(Dense(512, activation='relu'))
    print(model.output)
    model.add(Dropout(0.5))  # 随机失活
    print(model.output)

    # 全连接层2, Softmax评分, 输出nb_classese 个神经元
    model.add(Dense(nb_classes, activation='softmax'))
    print(model.output)
    exit()

    '''
    编译模型，配置模型的学习过程
    1.loss损失函数: 目标函数，可为预定义的损失函数
    2.optimizer优化器：参数可指定为预定义的优化器名，如rmsprop、adagrad或一个Optimizer类对象
    3.指标列表：对于分类问题，一般将该列表设置为metrics=['accuracy']
    '''
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    if verbose == True: print(model.summary())
    return model


def train(model, training_data, callback=True, batch_size=256, epochs=10):
    (x_train, y_train), (x_test, y_test), mapping, nb_classes = training_data
    print(mapping,nb_classes)
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    print("y_train",y_train.shape) # , "y_train",y_train[0])

    if callback == True:
        # Callback for analysis in TensorBoard
        tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    '''
    训练模型，放入批量样本，进行训练
    batch_size: 指定梯度下降时每个batch包含的样本数
    nb_epoch: 训练的轮数，nb指number of
    verbose: 日志显示， 0为不在标准输出流输出日志信息，1为输出进度条记录，2为epoch输出一行记录
    validation_data: 指定验证集
    fit 函数返回一个History的对象，其History.history属性记录了损失函数和其他指标的数值随epoch变化的情况，如果有验证集的话，也包含了验证集的这些指标变化情况
    '''
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[tbCallBack] if callback else None)

    # 评估模型，在测试集上评价模型的准确度
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # Offload model to file
    model_yaml = model.to_yaml()
    with open("bin/model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    save_model(model, 'bin/model.h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='A training program for classifying the EMNIST dataset')
    parser.add_argument('-f', '--file', type=str, help='Path .mat file data', required=True)
    parser.add_argument('--width', type=int, default=28, help='Width of the images')
    parser.add_argument('--height', type=int, default=28, help='Height of the images')
    parser.add_argument('--max', type=int, default=None, help='Max amount of data to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train on')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enables verbose printing')
    args = parser.parse_args()

    bin_dir = os.path.dirname(os.path.realpath(__file__)) + '/bin'
    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)

    training_data = load_data(args.file, width=args.width, height=args.height, max_=args.max, verbose=args.verbose)
    model = build_net(training_data, width=args.width, height=args.height, verbose=args.verbose)
    train(model, training_data, epochs=args.epochs)
