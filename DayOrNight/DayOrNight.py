import numpy as np
import numpy.random as rd

#This Neural Network is made to tell you if it is night or day based off of two inputs: the month and the time of day!

#Training Data (Generated)
def genData():
    sungraph = [ #goes like this: [sunrise, sunset] (index is month-1)
        [7.83,16.83], #jan
        [7.5, 17.5], #feb
        [7.85, 18.1], #mar
        [7, 19.75], #apr
        [6.15, 20.33], #may
        [5.51, 20.85], #jun
        [5.67, 21], #jul
        [6.1, 20.67], #aug
        [6.67, 19.85], #sep
        [7.25, 19], #oct
        [7.87, 18.15], #nov
        [7.5, 4.67], #dec
    ] #sungraph based on times in Toronto, Canada
    #For a better view of what this represents, go ahead and look at the picture titled sungraphVisual.jpg on this folder!

    data = []

    for i in range(50000):
        month = rd.randint(1,12) #generate a random month
        time = rd.randint(0,23) #generate a random time (in decimal hours)
        if time < sungraph[month-1][1] and time > sungraph[month-1][0]: #if it's within the sunset and sunrise
            daytimeness = 1; #daytime
        else:
            daytimeness = 0; #nighttime
        data.append([month, time, daytimeness]) #add to our datalist
    return data

tData = genData()
#network - for a better visual representation of the network look at the picture titled networkModel.jpg on this folder!
w1 = rd.randn()
w2 = rd.randn()
b = rd.randn()

def sigmoid(x): #keeps numbers between 0 and 1
    return 1/(1+np.exp(-x))

def sigmoid_p(x): #derivative of sigmoid
    return sigmoid(x) * (1-sigmoid(x))

#Training loop
learningRate = 0.2

for i in range(100000):
    ri = rd.randint(len(tData)) #get a random index
    point = tData[ri] #make a point that uses the random index

    z = (point[0] * w1) + (point[1] * w2) + b
    pred = sigmoid(z) #the actual prediction sqished into a sigmoid function

    target = point[2]
    cost = np.square(pred - target) #the squared cost (how off it is)
    dcost_pred = 2 * (pred - target) #the derivative of the squared cost

    dpred_dz = sigmoid_p(z) #the derivative of the prediction

    #derivatives of the weights and bias which is just what they are multiplied by
    dz_dw1 = point[0]
    dz_dw2 = point[1]
    dz_db = 1

    #multiplying the derivatives because the chain rule
    dcost_dw1 = dcost_pred * dpred_dz * dz_dw1 #derivative of the cost (for w1)
    dcost_dw2 = dcost_pred * dpred_dz * dz_dw2 #derivative of the cost (for w2)
    dcost_db = dcost_pred * dpred_dz * dz_db #derivative of the cost (for b)

    #taking from the parts of the neural network a fraction (learningRate) of the chained derivate above
    w1 = w1 - learningRate  * dcost_dw1
    w2 = w2 - learningRate * dcost_dw2
    b = b - learningRate * dcost_db


#Testing the network
print("January")
for i in range(0,24): #go through each hour
    z  = (1* w1) + (i * w2) + b
    print(i, sigmoid(z))

print("June") # do it again for june to (hopefully) see some difference in length of days
for i in range(0,24):
    z  = (6* w1) + (i * w2) + b
    print(i, sigmoid(z))

#For a first neural network that I had ever programmed in my life I feel pretty good about this one, despite it not being able to predict the sunset it can quite easily
#predict the sunrise and know that June has an earlier sunrise than January which was very nice to see! (Although quite the small difference it was still there)
#also it sometimes is complely wrong and I'm not 100% sure about that haha :/

#Thanks for viewing!
#Created by Dave Gershman w/ help from this video: https://www.youtube.com/watch?v=LSr96IZQknc
#7/31/2017
