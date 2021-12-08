# example of loading the mnist dataset
import numpy as np
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
import keras
from progress.bar import Bar
from decimal import *
import hardware


(trainX, trainy), (testX, testy) = mnist.load_data()
print('>Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('>Test: X=%s, y=%s' % (testX.shape, testy.shape))
     

def layer1(i, model):
    activation = testX[i].astype('float32')

    activation = activation/255.0
    #plt.imshow(activation)
    #plt.show()
    
    #layer 1: conv_0
    conv_0_weights = model.layers[0].get_weights()[0]
    conv_0_bias = model.layers[0].get_weights()[1]
    conv_0_shape = conv_0_weights.shape
    
    f1List = []
    for filterIndex in range(conv_0_shape[3]):
        layer = hardware.ConvEngine(activation, conv_0_weights[:,:,0,filterIndex],filterIndex)
        
        layer = layer + conv_0_bias[filterIndex]
        layer = layer * (layer>0)
        
        f1List.append(layer)

    f1 = np.array(f1List)
    #plt.imshow(f1[0])
    #plt.show()

    #layer 2: max pool

    f2List = []
    for featureIndex in range(f1.shape[0]):    
        layer = hardware.MaxPoolEngine(f1[featureIndex,:,:])
        
        f2List.append(layer)

    f2 = np.array(f2List)

    #layer 3: conv_1
    conv_1_weights = model.layers[2].get_weights()[0]
    conv_1_bias = model.layers[2].get_weights()[1]
    conv_1_shape = conv_1_weights.shape
    f3List = []
    for outputLayer in range(conv_1_shape[3]):
        partialSum = np.zeros((f2.shape[1]-conv_1_weights.shape[0]+1,f2.shape[1]-conv_1_weights.shape[0]+1))
        for inputLayer in range(conv_1_shape[2]):
            layer = hardware.ConvEngine(f2[inputLayer], conv_1_weights[:,:,inputLayer,outputLayer],inputLayer+outputLayer*conv_1_shape[2])
            partialSum += layer
        
        partialSum = partialSum + conv_1_bias[outputLayer]
        partialSum = partialSum * (partialSum>0)
        f3List.append(partialSum)

    f3 = np.array(f3List)
    #plt.imshow(f3[0])
    #plt.show()      

    #layer 4: conv_2
    conv_2_weights = model.layers[3].get_weights()[0]
    conv_2_bias = model.layers[3].get_weights()[1]
    conv_2_shape = conv_2_weights.shape

    newSize = f3.shape[1]-conv_2_shape[0]+1
    f4List = []
    for outputLayer in range(conv_2_shape[3]):
        partialSum = np.zeros((newSize,newSize))
        for inputLayer in range(conv_2_shape[2]):
            layer = hardware.ConvEngine(f3[inputLayer],conv_2_weights[:,:,inputLayer,outputLayer],inputLayer+outputLayer*conv_1_shape[2])
            partialSum += layer

        partialSum = partialSum + conv_2_bias[outputLayer]
        partialSum = partialSum * (partialSum>0)
        f4List.append(partialSum)

    f4 = np.array(f4List)
    #plt.imshow(f4[0])
    #plt.show()

    #layer 5
    
    f5List = []
    for featureIndex in range(f4.shape[0]):
        layer = hardware.MaxPoolEngine(f4[featureIndex,:,:])
        
        f5List.append(layer)

    f5 = np.array(f5List)
    #plt.imshow(f5[0])
    #plt.show()
 
    #layer 6
    
    f6List = []
    
    for row in range(f5.shape[1]):
        for column in range(f5.shape[2]):
            for inputFeature in range(f5.shape[0]):
                f6List.append(f5[inputFeature,row,column])

    f6 = np.array(f6List)
     
    #layer 7: Dense
    dense_0_weights = model.layers[6].get_weights()[0]
    dense_0_bias = model.layers[6].get_weights()[1]
    
    f7 = np.dot(f6,dense_0_weights) + dense_0_bias
    f7 = f7 * (f7>0)

    #layer 8: Dense_1
    dense_1_weights = model.layers[7].get_weights()[0]
    dense_1_bias = model.layers[7].get_weights()[1]
    
    f8 = np.dot(f7,dense_1_weights) + dense_1_bias
    f8 = f8 * (f8>0)
    f8_out = np.exp(f8)/sum(np.exp(f8))
    return (np.argmax(f8_out))

def testAccuracy(tests):
    correct = 0
    
    model=keras.models.load_model('../models/final_model.h5')
    with Bar('Processing...', max = tests) as bar:
        for i in range(tests):
            prediction = layer1(i, model)
            bar.next()
            if prediction == testy[i]:
                correct+=1

    print(str(correct) + "/" + str(tests))

    accuracy = correct/tests
    return accuracy


def init():
    model=keras.models.load_model('../models/final_model.h5')
    
    conv_0_weights = model.layers[0].get_weights()[0]
    conv_0_bias = model.layers[0].get_weights()[1]
    conv_0_shape = conv_0_weights.shape
    activation = testX[0]
    activation = activation/255
    f1List = []
    f1ListEngine = []
    for filterIndex in range(conv_0_shape[3]):
        layer = hardware.verifyConvEngine(activation, conv_0_weights[:,:,0,filterIndex])
        engineLayer = hardware.ConvEngine(activation,conv_0_weights[:,:,0,filterIndex],filterIndex)
        
        layer = layer + conv_0_bias[filterIndex]
        engineLayer = engineLayer + conv_0_bias[filterIndex]
        
        layer = layer * (layer>0)
        engineLayer = engineLayer * (engineLayer>0)
        
        f1List.append(layer)
        f1ListEngine.append(engineLayer)

    f1 = np.array(f1List)
    f1Engine = np.array(f1ListEngine)
    #print(np.max(f1))
    print(f1.shape)
    print(f1Engine.shape)
    plt.imshow(f1[0])
    plt.show()
    plt.imshow(f1Engine[0])
    plt.show()



testAccuracy(5)

