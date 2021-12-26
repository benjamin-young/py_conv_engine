import struct
from unittest import result
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy import signal
from skimage.measure import block_reduce
from random import randint

def ZeroPad(image,mask):
    maskSize = len(mask)
    imageSize = len(image)
    result = np.zeros((imageSize+maskSize-1,imageSize+maskSize-1))
    for i in range(len(image)):
        for j in range(len(image)):
            result[i][j] = image[i][j]

    return result

def bitFlip(num):
    #print(num)
    scaledNum = int(round(abs(num)*(2**8)))

    fixed_bits = "{0:b}".format(scaledNum)
    #print(fixed_bits)
    binaryOutIter = ['0'] * (16-len(fixed_bits))
    
    #if negative add signed bit
    if(num<0):
        binaryOutIter[0] = '1'

    binaryOutIter += fixed_bits
    #print(binaryOutIter)

    #random bitflip position in the 16 bit number
    randPos = randint(0,15)
    
    if(binaryOutIter[randPos] == '1'):
        binaryOutIter[randPos] = '0'
    else:
        binaryOutIter[randPos] = '1'   
    
    #print(binaryOutIter)
    binaryOut = ('').join(binaryOutIter[1:16])
    #print(binaryOut)
    #binaryOutFlipped = ''.join('1' if x == '0' else '0' for x in binaryOut)
    flippedNum = int(binaryOut,2)
    if(binaryOutIter[0]=='1'):
        flippedNum=flippedNum*(-1)
    scaledFilppedNum = flippedNum/(2**8)
    #print(scaledFilppedNum)
    return scaledFilppedNum

def ConvEngine(image, kernal, filterNum, array):
    result = []
    pArray = []
    mask = np.zeros((len(kernal),len(kernal)))
    pSum=0

    peArrayRows = len(array.conv_array)
    rowIndex = filterNum % peArrayRows

    for row in range(len(image)+1-len(mask)):
        
        for column in range(len(image[0])+1-len(mask)):
            mask = image[row:row+len(mask),column:column+len(mask[0])]

            #perform repair and fault injection 
            array.timeStep()
            
            for kernalRow in range(len(kernal)):
                for kernalColumn in range(len(kernal[0])):
                    maskReg = mask[kernalRow][kernalColumn]
                    filterReg = kernal[kernalRow][kernalColumn]

                    pe = len(pArray)

                    mulResult = fixed_point_multiplication(float_to_fixed(maskReg,8),float_to_fixed(filterReg,8),8)

                    #if true use correct answer
                    if(array.operate(rowIndex, pe)):
                        peResult = mulResult
                    #if false use non redundant PE output
                    else:
                        #no fault in the PE
                        if(array.conv_array[rowIndex][pe] == 0):
                            peResult = mulResult
                        #fault in the PE
                        else:
                            #print("fault")
                            peResult = bitFlip(mulResult)
                            

                    pArray.append(peResult)
                    
                    
            for p in pArray:
                pSum += p

            result.append(pSum)
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


def float_to_fixed(float_num,fractional_bits):
    scalingFactor = 1/(2**fractional_bits)
    temp = float_num / scalingFactor
    fixed_int = int(round(temp))
    fixed_bits = "{0:b}".format(fixed_int)
    scaled_fixed_int = fixed_int * scalingFactor
    
    return scaled_fixed_int

def fixed_point_multiplication(a,b,fractional_bits):
    tempA = int(round(a * (2**fractional_bits)))
    tempB = int(round(b * (2**fractional_bits)))

    tempResult = tempA*tempB
    
    scaledTempResult = round(tempResult/(2**fractional_bits))
    result = scaledTempResult/(2**fractional_bits)
    
    return result

    
