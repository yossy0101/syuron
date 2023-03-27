import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD
import os
import sys
sys.path.append(os.path.abspath("../common"))
from data_load import data_load
from model import make_model
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

parser = argparse.ArgumentParser(description='Keras CNN')
parser.add_argument('-image_size', default=256, type=int)
parser.add_argument('-batch_size', default=64, type=int)
parser.add_argument('-epochs', default=2, type=int)
parser.add_argument('-datasets', help='choices detasets', default='cifar10', 
                    choices=['cifar10', 'cifar100', 'food101', 'tiny', 'imagenet100', 'imagenet'])
parser.add_argument('-use_model', help='choices model', default='vgg16', 
                    choices=['vgg16', 'vgg19', 'resent18', 'resnet34', 'resnet50', 'resnet101'])

def step_decay(epoch):
  x = 0.1
  if epoch >= 100: x = 0.01
  if epoch >= 150: x = 0.001
  return x

def allocate_gpu_memory(gpu_number=0):
  print('gpuの使用量抑えてくれーー！！！')
  physical_devices = tf.config.experimental.list_physical_devices('GPU')

  if physical_devices:
    try:
      print("Found {} GPU(s)".format(len(physical_devices)))
      tf.config.set_visible_devices(physical_devices[gpu_number], 'GPU')
      tf.config.experimental.set_memory_growth(physical_devices[gpu_number], True)
      print("#{} GPU memory is allocated".format(gpu_number))
    except RuntimeError as e:
      print(e)
  else:
    print("Not enough GPU hardware devices available")
    
def main():
  allocate_gpu_memory()
  args = parser.parse_args()
  lr_decay = LearningRateScheduler(step_decay)
  loader, image_num, defo_image_size, num_classes = data_load(args.image_size, args.batch_size, args.datasets)
  model = make_model(args.use_model, defo_image_size, args.image_size, num_classes)
  model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01,momentum=0.9,nesterov=True), metrics=["accuracy"])
  # model.summary()
  model.fit_generator(loader['train'], validation_data=loader['test'], epochs=args.epochs, callbacks=[lr_decay])
  
if __name__ == '__main__':
  main()