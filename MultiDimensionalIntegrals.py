import numpy as np
import matplotlib.pyplot as plt
import time

N = 25000
x = np.zeros(N)
values = np.zeros(N)

x[0] = 0.5
AcceptanceProb = np.copy(values)

def AutoCovariance(f,s,mu,N):
    return (1/(N-s))*np.dot(f[0:N-s],f[s:N])

def ProbabilityDensity(x):
    return np.exp(-(x**2)/2)

def function(x):
    return x**2

Delta = 1.5
starttime = time.clock()
for i in range(len(x)-1):
    #Sample U from uniform
    u = np.random.random()
    #Sample xd from a proposal distribution a sample point (needs to be symmetric dist)
    xd = (2*np.random.random()-1)*Delta + x[i]
    #Add the function value in the list
    values[i] = function(x[i])
    #AceptanceProb
    AcceptanceProb[i] = min(1,ProbabilityDensity(xd)/ProbabilityDensity(x[i]))
    #Do the Metropolis Part
    if u < AcceptanceProb[i]:
        x[i+1] = xd
    else:
        x[i+1] = x[i]

print("Integral = " + str(np.mean(values)) + " give or take " + str(2*np.sqrt(np.var(values)/N)))
print("Acceptance probability = " + str(np.mean(AcceptanceProb)))
print("This took " + str(time.clock()-starttime) + " seconds.")

a = time.clock()
R = [AutoCovariance(values,i,np.mean(values),len(values)) for i in range(int(len(values)/100))]
print(time.clock()-a)
plt.plot(R)
plt.show()

