"""
Tested with python 3. 
"""

import math
import cmath
import random
import numpy as np
import pylab
import matplotlib.pyplot as plt

pi2j = cmath.pi*2j

def convol(a, b, q):
    #convolution of two vectors of dimension q
    c = np.array( [ 0.0 for i in range(q)] )
    for i in range(q):
        for j in range(q):
            c[i]+= a[(i-j)%q]*b[j]
    return c

def FT(f, q, b):
    #input: f, output, hat f over Z_q
    FT_array = np.array( [ 0.0 for i in range(q)] )
    for z in range(q):
        Wz = 0
        for x in range(q):
            Wz += f[x] * cmath.exp(pi2j*float(x*z)/q) / math.sqrt(q) 
        if b ==1:
            FT_array[z] = Wz
        if b==2:
            FT_array[z] = abs(Wz)
    return FT_array


def ComplexGauss(x, r, s, c):
    """  exp(-pi (1/r^2 + i/s^2) (x-c)^2 )  """
    x2 = float( (x-c)*(x-c) )
    return cmath.exp( -cmath.pi*x2*(1.0/(r*r)+1.0j/(s*s) ) )

def cg(r, s, c, q):
    """  plotting complex Gaussian with r, s, center c, and modulus q, and its DFT mod q  """
    cg = np.array( [ ComplexGauss(x, r, s, c) for x in range(q)] )
    FTcg = FT(cg, q,1)

    print("cg, with r, s, c, q = ", r,s, c, q)
    print("s^2 r^4/(s^4+r^4) = ", (s**2) * (r**4)*1.0/(s**4+r**4)  )

    plt.plot( range(0, q), cg )
    plt.show()
    plt.close()

    plt.plot( range(0, q), FTcg )
    plt.show()
    plt.close()

    return cg

cg(54, 27.5, 100, 200)
#cg(100000, 2.82842, 0, 160)
