from scipy.stats import poisson
from random import randint

class PE_Array:
    conv_array = []
    duplicated_conv_array = []
    redundant_array = []
    redundant_array_faulty = []
    distSize = 10000
    #SEU probability
    rate = 0
    faultDist = poisson.rvs(rate, size=distSize)

    #make arrays the right size
    def __init__(self, rows, kernalSize, redundantPEs):

        self.conv_array = [ [0]*kernalSize for i in range(rows)]
        self.duplicated_conv_array = self.conv_array
        
        self.redundant_array = [0]*redundantPEs
        self.redundant_array_faulty = [0]*redundantPEs
        

    def repair(self):
        self.conv_array = [ [0]*len(self.conv_array[0]) for i in range(len(self.conv_array))]
        self.duplicated_conv_array = self.conv_array
        
        self.redundant_array = [0]*len(self.redundant_array)
        self.redundant_array_faulty = [0]*len(self.redundant_array)

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
                        return False
                    
            #if there are no redundant PEs one of the generated results will have to be chosen 
            return False
            
        else:
            #there are no faults in the PE or duplicated PE therefore the outputs would be the
            #same, therefore use correct result
            return True

    def updateDistribution(self, rate):
        self.faultDist = poisson.rvs(rate, size=self.distSize)