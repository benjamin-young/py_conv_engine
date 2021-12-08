# py_conv_engine
code for analysing the effect of single event upsets in a convolution engine on the accuracy of the convolutional neural network inference.

# Convolution Engine Design
![conv_engine](https://user-images.githubusercontent.com/8210731/145269247-fe16a1ad-f258-41fc-990c-fd52d06e90df.png)

# Fault Tolerance Design 
Each PE in the convolution engine is duplicated so that faults can be detected.
1. When a fault is detected the inputs are redirected to a PE in a bank of redundant PEs
2. The output which the redundant PE agrees with is chosen as the output
3. If there are no more redundant PEs left one of the pool, one of the original PE outputs is chosen

# Fault injection
Faults are modelled as SEUs, which only appear for a single time step, that alter the functionality of a PE. The number of SEUs in a timestep is given by a Poisson distribution.
