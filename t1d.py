import math
import sympy as sym
import numpy as np
import scipy as sp
import scipy.linalg
import symarray_helper

I = sym.symbols("I", commutative=False)
x = sym.symbols("X", commutative=False)
y = sym.symbols("Y", commutative=False)
c = sym.symbols("C")

# Simple test of rules in arxiv:1003.1047

einsum = symarray_helper.einsum

def mdot(*args):
    """
    multiply out a sequence of matrices
    """
    ret = args[0]
    for a in args[1:]:
        ret = np.dot(ret, a)
    return ret

def test_sq():

    L = 2
    T = np.zeros([L,L,4,4,4,4], dtype=object)

    # Tensors have input and output arrow, left->right, down->up
    a = 0 # input index, arrow to right, or arrow up
    z = 3 # output index, arrow to right, or arrow up

    b = 1 # x is being transmitted to right and/or up
    c = 2 # x is being transmitted up, to meet x transmitted from left
    
    for i in range(2):
        for j in range(2):
            T[i,j,a,a,a,a] = 1#sym.symbols("I"+str(i)+str(j),commutative = False)
            
            T[i,j,a,a,a,b] = sym.symbols("x"+str(i)+str(j),commutative = False) # 1 
            T[i,j,a,b,a,z] = sym.symbols("x"+str(i)+str(j),commutative = False) # 1
            T[i,j,c,a,a,a] = sym.symbols("x"+str(i)+str(j),commutative = False) # 1
            T[i,j,b,a,a,a] = sym.symbols("x"+str(i)+str(j),commutative = False) # 
            
            T[i,j,z,a,b,a] = sym.symbols("x"+str(i)+str(j),commutative = False) # 2
            T[i,j,b,b,a,a] = sym.symbols("c",commutative = True) # 2
            T[i,j,a,b,a,b] = sym.symbols("c",commutative = True) # 2 
            T[i,j,b,a,b,a] = sym.symbols("c",commutative = True) # 2 
            T[i,j,z,b,c,a] = sym.symbols("c",commutative = True) # 3 
            T[i,j,a,b,a,z] = sym.symbols("x"+str(i)+str(j),commutative = False) # 4
            

    TX0 = einsum("uldI,UIDR->uldDRU", T[0,0,:,:,:,:], T[0,1,:,:,:,:])
    TX1 = einsum("uldI,UIDR->uldDRU", T[1,0,:,:,:,:], T[1,1,:,:,:,:])
    TSQ = einsum("ulijRU,imeESj->ulmeESRU", TX1, TX0)

    TSQ.dump("tsq_file")
    print TSQ[a,a,a,a,a,z,a,a] # should be X00 * X01
    print TSQ[a,a,a,a,a,a,a,z] # should be cX00*X11 + cX10*X01 + X01*X11
    print TSQ[a,a,a,a,a,a,z,a] # should be X10 * X11 
    print TSQ[z,a,a,a,a,a,a,a] # should be X10 * X00

def test_2x3():
    # Index convention: counter clockwise, from leftmost upper index
    # e.g.
    #
    #   0              0    5
    #   |              |    |
    #1 --- 3       1------------4    etc. 
    #   |              |    |
    #   2              2    3 

    
    T = np.zeros([2,3,4,4,4,4], dtype=object)

    # Tensors have input and output arrow, left->right, down->up

    a = 0 # input index, arrow to right, or arrow up
    z = 3 # end index, arrow to right, or arrow up

    b = 1 # x is being transmitted to right and/or up
    c = 2 # x is being transmitted up, to meet x transmitted from left

    for i in range(2):
        for j in range(3):
            T[i,j,a,a,a,a] = 1
            T[i,j,a,z,a,z] = 1

            T[i,j,a,a,a,b] = sym.symbols("x"+str(i)+str(j)) # 1 
            T[i,j,a,b,a,z] = sym.symbols("x"+str(i)+str(j)) # 1
            T[i,j,c,a,a,a] = sym.symbols("x"+str(i)+str(j)) # 1
            T[i,j,b,a,a,a] = sym.symbols("x"+str(i)+str(j)) #
            
            T[i,j,z,a,b,a] = sym.symbols("x"+str(i)+str(j)) # 2
            T[i,j,b,b,a,a] = sym.symbols("c") # 2 

            T[i,j,a,b,a,b] = sym.symbols("c") # 2 
            T[i,j,b,a,b,a] = sym.symbols("c") # 2 

            T[i,j,z,b,c,a] = sym.symbols("c") # 3 
            T[i,j,a,b,a,z] = sym.symbols("x"+str(i)+str(j)) # 4


    # row 0 
    TX0 = einsum("uldI,UIDR->uldDRU", T[0,0,:,:,:,:], T[0,1,:,:,:,:])
    TX0 = einsum("uldDIU,vIer->uldDervU", TX0, T[0,2,:,:,:,:])

    # row 1
    TX1 = einsum("uldI,UIDR->uldDRU", T[1,0,:,:,:,:], T[1,1,:,:,:,:])
    TX1 = einsum("uldDIU,vIer->uldDervU", TX1, T[1,2,:,:,:,:])

    # contract down
    TSQ = einsum("ulIJKrvU,ILdDERKJ->ulLdDERrvU", TX1, TX0)
    TSQ.dump("tsq_file")
    
    # Total is 15 terms (6 C 2) -- I think this is right
    term1= TSQ[a,a,a,a,a,a,z,a,a,a] # 1st row
    term2= TSQ[a,a,a,a,a,a,a,z,a,a] # 2nd row
    term3= TSQ[z,a,a,a,a,a,a,a,a,a] # 1st col
    term4= TSQ[a,a,a,a,a,a,a,a,a,z] # 2nd col
    term5= TSQ[a,a,a,a,a,a,a,a,z,a] # 3rd col

    print term1+term2+term3+term4+term5

    # Note that in the above, it is necessary to specify the final output index all along the top and right
    # hand side. We can contract over all the above cases, by adding an additional row/col of tensors, that makes
    # sure that only a single bond == z, on the top or right
    
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
    
