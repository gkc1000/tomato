import math
import sympy as sym
import numpy as np

I = sym.symbols("I", commutative=False)
x = sym.symbols("x", commutative=False)
y = sym.symbols("y", commutative=False)
c = sym.symbols("c")

# Simple test of rules in arxiv:1003.1047

def mdot(*args):
    """
    multiply out a sequence of matrices
    """
    ret = args[0]
    for a in args[1:]:
        ret = np.dot(ret, a)
    return ret

def test_nn():
    """
    Nearest neighbour
    """
    T = np.zeros([3,3], dtype=object)
    T[0,0] = I
    T[0,1] = x
    T[1,2] = y
    T[2,2] = I

    H =  mdot(T,T,T,T)
    for i in range(3):
        for j in range(3):
            print (i, j), H[i,j]

    aux_bra = np.zeros([3])
    aux_bra[0] = 1.

    aux_ket = np.zeros([3])
    aux_ket[0] = 1.
    aux_ket[2] = 1.
    
    print np.dot(aux_bra, np.dot(H, aux_ket))
    
def test_exp():
    """
    Exp decay
    """
    T = np.zeros([3,3], dtype=object)
    T[0,0] = I
    T[0,1] = x
    T[1,1] = c
    T[1,2] = c*y
    T[2,2] = I

    print mdot(T,T,T)[0,2]
