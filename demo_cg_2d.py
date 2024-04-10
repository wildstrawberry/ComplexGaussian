"""
Tested with python 3. 
"""

import math
import cmath
import random
import numpy as np
import pylab
import matplotlib.pyplot as plt
import time

pi2j = cmath.pi*2j

def plotI(I, col):
    plt.imshow(I.real, cmap=col, interpolation='none')
    plt.colorbar()
    plt.show()
    plt.close()

def plotsqu(I, col):
    plt.imshow((I.real)**2+(I.imag)**2, cmap=col, interpolation='none')
    plt.colorbar()
    plt.show()
    plt.close()


def SolveLWEdemo():

    b_1 = -1
    b_2 = 2
    D = 1
    c = 4
    u2 = D*D*(b_1**2+b_2**2)   # u2 = u^2, similarly for t2, r2, s2
    t2 = c*u2
    tx = u2+t2
    r = 380.0
    r2 = r**2
    s2 = 3.1255*u2*t2 # s^2 need to be set manually
    P = tx*tx*2
    M = tx*2  
    fin = M*u2
    z_1 = tx**2
    z_2 = tx**2-100
    y_2 = - P-400
    conds1 = s2*(r**4)*t2/(u2*(r**4+s2**2)*(tx)**2)  # used to test whether s is good
    print( "M = ", M, " b= (", b_1, b_2, "), D = ", D, ", t^2 = ", t2, ", M/2 = ", tx, ", r = ", r, "s^2 = ", s2, "sigma = ", s2*2/r, "cond.s1_checker: (good if very close to 2)", conds1)

    # Step 1
    
    time0 = time.perf_counter()
    
    I1 = np.array( [[ 0.0j for i in range(P)] for j in range(P) ] )
    ## x_1: vertcal, x_2: horizontal, 
    for x_1 in range(P):
        for x_2 in range(P):
            if b_2*x_1 +x_2  == - y_2: 
                I1[x_1][x_2] = cmath.exp( -cmath.pi*( 1/r2 + 1.0j/s2 )*((x_1)**2+(x_2)**2) )


    # Step 2
    I2 = np.fft.fft2(I1)

    # Step 3
    time2 = time.perf_counter()

    I3 = np.array( [[ I2[i][j] for j in range(P)] for i in range(P) ] )
    for x_1 in range(P):
        for x_2 in range(P):
            I3[x_1][x_2] *= cmath.exp( -cmath.pi*( t2*r2*s2*(s2 - r2*1.0j)/(P*P*u2*(r**4+s2**2)) )*((x_1 - z_1)**2+(x_2 - z_2)**2) )

    # Step 4
    I4 = np.fft.fft2(I3)

    # Step 5
    time4 = time.perf_counter()
    I5 = np.array( [[ I4[(i*tx)%P][j*tx] for j in range(M)] for i in range(M) ] )

    # Step 6 amp
    I6 = np.fft.fft2(I5)
    # Step 6 squared
    I6a = np.array( [[ I6[(i+z_1)%M][(j+z_2-y_2)%M] for j in range(M)] for i in range(M) ] )
    time6 = time.perf_counter()

    print( "time 6 - time 0:", time6 - time0, "time 2 - time 0:", time2 - time0, "time 4 - time 2:", time4 - time2 )
    
    I7lwesim = np.array( [[ 0.0+0.0j for j in range(M)] for i in range(M) ] )
    
    I7rate = np.array( [[ 0.0+0.0j for j in range(M)] for i in range(M) ] )
    
    kprime = int((z_1*(-1)+z_2*b_2)*D/tx - (y_2*b_2*D)/u2)
    print("kprime = ", kprime)

    for j in range(M):
        for k in range(2):
            x_1 = (2*D*j - b_2*D*k)*(-D) + kprime*(-D)
            x_2 = (2*D*j - b_2*D*k)*b_2*D + tx*k + kprime*b_2*D
            I7lwesim[x_1%M][x_2%M] = cmath.exp( -pi2j*( (2*D*j - b_2*D*k)**2/(2*M) - k**2/4.0) )  # create the simulated I7
            I7rate[x_1%M][x_2%M] = I6a[x_1%M][x_2%M]/I7lwesim[x_1%M][x_2%M]  # used to test whether the simulation is correct

    I7c = np.array( [[ I7lwesim[i][j]*cmath.exp(pi2j* (i*i)/(2*M) ) for j in range(M)] for i in range(M) ] )
    if D>1:
        I7a = np.array( [[ I7lwesim[i%M][j%M] for j in range(M*D)] for i in range(M*D) ] )
        I7b = np.array( [[ I7a[i*D][j*D] for j in range(M)] for i in range(M) ] )
        I7c = np.array( [[ I7b[i][j]*cmath.exp(pi2j* (i*i)/(2*M) ) for j in range(M)] for i in range(M) ] )
    
    I7d = np.fft.fft2(I7c) 

    plotI(I1,'RdBu')
    plotI(I2,'RdBu')
    plotI(I3,'RdBu')
    plotI(I4,'RdBu')
    plotI(I5,'RdBu')
    plotI(I6,'RdBu')
    print("I7 simulated:")
    plotI(I7lwesim,'RdBu')
    print("I7 real/I7 simulated: (correct if it is 1 on the support, 0 elsewhere)")
    plotsqu(I7rate,'Blues')
    plotI(I7c,'RdBu')
    plotI(I7d,'RdBu')

SolveLWEdemo()

