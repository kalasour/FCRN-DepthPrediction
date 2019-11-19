import argparse
import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from matplotlib import pyplot as plt
from PIL import Image
import glob
import models

def predict(model_data_path, image_path):

    
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1
   
    # Read image
    
   
    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)
        
    with tf.Session() as sess:
        print('Loading model...')
        net.load(model_data_path, sess) 
        testfileloc = list(glob.glob(image_path+'\input\*'))
        for i in testfileloc:
            print('predicting...',i)
            img = Image.open(i)
            img = img.resize([width,height], Image.ANTIALIAS)
            img = np.array(img).astype('float32')
            img = np.expand_dims(np.asarray(img), axis = 0)
            pred = sess.run(net.get_output(), feed_dict={input_node: img})
            fig = plt.figure()
            ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
            fig.colorbar(ii)
            # plt.show()
            print('saved..',i.replace('input','output'))
            plt.savefig(i.replace('input','output'))

        
        
        return pred
        
                
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_paths', help='Directory of images to predict')
    args = parser.parse_args()

    # Predict the image
    pred = predict(args.model_path, args.image_paths)
    
    os._exit(0)

if __name__ == '__main__':
    main()

        



