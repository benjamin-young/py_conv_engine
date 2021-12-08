from PE_Array import PE_Array
import struct
import numpy as np
from matplotlib import pyplot as plt
import math

rows = 8
kernal_size = 9
redundantPEs = 8

array = PE_Array(rows,kernal_size,redundantPEs)

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
    rowIndex = filterNum % peArrayRows

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
from scipy import signal
from skimage.measure import block_reduce

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