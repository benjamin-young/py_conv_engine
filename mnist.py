# example of loading the mnist dataset
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
from scipy import signal
import math
from skimage.measure import block_reduce
import keras
from progress.bar import Bar
import struct
from decimal import *
from scipy.stats import poisson
from random import randint

(trainX, trainy), (testX, testy) = mnist.load_data()
print('>Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('>Test: X=%s, y=%s' % (testX.shape, testy.shape))

faultCount = 0
faultyRedundantPE = 0

class PE_Array:

    conv_array = []
    duplicated_conv_array = []
    redundant_array = []
    redundant_array_faulty = []
    distSize = 10000
    #SEU probability
    probability = 0.1
    faultDist = poisson.rvs(probability, size=distSize)

    #make arrays the right size
    def __init__(self, rows, kernalSize, redundantPEs):
        self.conv_array = [ [0]*kernalSize for i in range(rows)]
        self.duplicated_conv_array = self.conv_array
        
        self.redundant_array = [0]*redundantPEs
        self.redundant_array_faulty = [0]*redundantPEs
        print(self.redundant_array)


    def repair(self):
        self.conv_array = [ [0]*len(self.conv_array[0]) for i in range(len(self.conv_array))]
        self.duplicated_conv_array = self.conv_array
        
        self.redundant_array = [0]*redundantPEs
        self.redundant_array_faulty = [0]*redundantPEs
    
    def faultInjection(self):
        distIndex = randint(0,9999)
        faults = self.faultDist[distIndex]

        conv_array_size = len(self.conv_array)*len(self.conv_array[0])
        totalPEs = conv_array_size*2 + len(self.redundant_array)

        for i in range(faults):
            #FIX to include redundant PEs
            randIndex = randint(0, totalPEs -1)
            if(randIndex<conv_array_size):
                randRow = randint(0, len(self.conv_array)-1)
                randPE = randint(0, len(self.conv_array[0])-1)
                self.conv_array[randRow][randPE] = 1
            elif(randIndex<2*conv_array_size):
                randRow = randint(0, len(self.conv_array)-1)
                randPE = randint(0, len(self.conv_array[0])-1)
                self.conv_array[randRow][randPE] = 1
            else:
                #print("faulty redundant PE")
                randPE = randint(0, len(self.redundant_array)-1)
                self.redundant_array_faulty[randPE] = 1

    def timeStep(self):
        #print("tik")
        
        self.repair()
        self.faultInjection()

    #returns true for give correct answer and false for guess conv_array answer
    def operate(self,row,pe):
        #there is a fault in one of the PEs therefore output cannot be trusted
        if(self.conv_array[row][pe] == 1 or self.duplicated_conv_array[row][pe] == 1):
            #check if there is a free redundant PE
            #0 if free, 1 if taken

            #if there is a redundant PE use correct answer
            for i in range(len(self.redundant_array)):
                if(self.redundant_array[i] == 0):
                    #mark PE as used
                    self.redundant_array[i]=1

                    #check if redundantPE is faulty
                    if(self.redundant_array_faulty[i] == 0):
                        
                        return True
                    else:
                        global faultyRedundantPE
                        faultyRedundantPE+=1
                        return False
                    
            #if there are no redundant PEs one of the generated results will have to be chosen 
            return False
            
        else:
            #there are no faults in the PE or duplicated PE therefore the outputs would be the
            #same, therefore use correct result
            return True
            

def ZeroPad(image,mask):
    maskSize = len(mask)
    imageSize = len(image)
    result = np.zeros((imageSize+maskSize-1,imageSize+maskSize-1))
    for i in range(len(image)):
        for j in range(len(image)):
            result[i][j] = image[i][j]

    return result

def bitFlip(num):
    binaryOut = format(struct.unpack('!I', struct.pack('!f', num))[0], '016b')
    binaryOutFlipped = ''.join('1' if x == '0' else '0' for x in binaryOut)
    flippedFloat = struct.unpack('!f',struct.pack('!I', int(binaryOutFlipped, 2)))[0]
    inputFloat = struct.unpack('!f',struct.pack('!I', int(binaryOut, 2)))[0]

    return flippedFloat

def ConvEngine(image, kernal, filterNum):
    result = []
    pArray = []
    mask = np.zeros((len(kernal),len(kernal)))
    #print(filterNum)
    pSum=0

    peArrayRows = len(array.conv_array)
    rowIndex = filterNum%peArrayRows

    #print(rowIndex)
    
    for row in range(len(image)+1-len(mask)):
        #print(row)
        for column in range(len(image[0])+1-len(mask)):
            mask = image[row:row+len(mask),column:column+len(mask[0])]

            #perform repair and fault injection 
            array.timeStep()
            
            for kernalRow in range(len(kernal)):
                for kernalColumn in range(len(kernal[0])):
                    maskReg = np.float16(mask[kernalRow][kernalColumn])
                    filterReg = np.float16(kernal[kernalRow][kernalColumn])

                    pe = len(pArray)

                    #if true use correct answer
                    if(array.operate(rowIndex, pe)):
                        peResult = np.float16(maskReg*filterReg)
                    #if false use non redundant PE output
                    else:
                        #no fault in the PE
                        if(array.conv_array[rowIndex][pe] == 0):
                            peResult = np.float16(maskReg*filterReg)
                        #fault in the PE
                        else:
                            #print("fault")
                            peResult = bitFlip(np.float16(maskReg*filterReg))
                            global faultCount
                            faultCount+=1

                    pArray.append(peResult)
                    
                    
            for p in pArray:
                pSum += p

            #pSum = bitFlip(pSum)

            result.append(np.float16(pSum))
            pSum=0
            pArray = []

    resultArray = np.array(result)
    width = int(math.sqrt(len(resultArray)))
    resultArray = np.reshape(resultArray, (width,width))

    return resultArray

def verifyConvEngine(image, kernal):
    f1 = signal.correlate2d(image, kernal, mode='valid')
    return f1
    

def MaxPoolEngine(feature_map):
    result = []
    mask = np.zeros((2,2))
    poolNumber = len(feature_map)/2
    poolNumber = math.floor(poolNumber)
    #print(poolNumber)
    
    for row in range(0,poolNumber*2,2):
        line = []
        for column in range(0,poolNumber*2,2):
            
            mask = feature_map[row:row+len(mask),column:column+len(mask[0])]
            #print("mask: " +str(mask.shape))
            maxValue = mask.max()
            line.append(maxValue)
        lineArray = np.array(line)
        result.append(lineArray)
    
    resultArray = np.array(result)
    return resultArray

def verifyMaxPoolEngine(feature_map):
    max_pool_2d = block_reduce(feature_map, (2,2), np.max)
    plt.suptitle('maxpool verify output')
    plt.imshow(max_pool_2d)
    plt.show()

def streamToMap(feature_stream):    
    count=0
    width = int(math.sqrt(len(feature_stream)))
    ResultImage = np.zeros((width,width))
    for i in range(width):
        for j in range(width):
            ResultImage[i][j] = feature_stream[count]
            count+=1
    return ResultImage


def FullyConnectedEngine(feature_map,weights,biases):
    result = feature_map.dot(weights) + biases
    result = result * (result>0)
    return result
    

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
        layer = ConvEngine(activation, conv_0_weights[:,:,0,filterIndex],filterIndex)
        
        layer = layer + conv_0_bias[filterIndex]
        layer = layer * (layer>0)
        
        f1List.append(layer)

    f1 = np.array(f1List)
    #plt.imshow(f1[0])
    #plt.show()

    #layer 2: max pool

    f2List = []
    for featureIndex in range(f1.shape[0]):    
        layer = MaxPoolEngine(f1[featureIndex,:,:])
        
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
            layer = ConvEngine(f2[inputLayer], conv_1_weights[:,:,inputLayer,outputLayer],inputLayer+outputLayer*conv_1_shape[2])
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
            layer = ConvEngine(f3[inputLayer],conv_2_weights[:,:,inputLayer,outputLayer],inputLayer+outputLayer*conv_1_shape[2])
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
        layer = MaxPoolEngine(f4[featureIndex,:,:])
        
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
    
    model=keras.models.load_model('final_model.h5')
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
    model=keras.models.load_model('final_model.h5')
    
    conv_0_weights = model.layers[0].get_weights()[0]
    conv_0_bias = model.layers[0].get_weights()[1]
    conv_0_shape = conv_0_weights.shape
    activation = testX[0]
    activation = activation/255
    f1List = []
    f1ListEngine = []
    for filterIndex in range(conv_0_shape[3]):
        layer = verifyConvEngine(activation, conv_0_weights[:,:,0,filterIndex])
        engineLayer = ConvEngine(activation,conv_0_weights[:,:,0,filterIndex],filterIndex)
        
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


rows = 8
kernal_size = 9
redundantPEs = 3

array = PE_Array(rows,kernal_size,redundantPEs)

testAccuracy(5)
print("faulty count:  " + str(faultCount))

print("faulty redundant PEs: "+ str(faultyRedundantPE))
