import math
import sympy as sym
import numpy as np
import scipy as sp
import scipy.linalg
import symarray_helper

# I = sym.symbols("I", commutative=False)
# x = sym.symbols("X", commutative=False)
# y = sym.symbols("Y", commutative=False)
# c = sym.symbols("C")

a = 0 # input index, arrow to right, or arrow up
z = 3 # end index, arrow to right, or arrow up
z = a
b = 1 # x is being transmitted to right and/or up
c = 2 # x is being transmitted up, to meet x transmitted from left
d = 4 # for boundary

indices = {}
indices[a] = "a"
indices[z] = "z"
indices[b] = "b"
indices[c] = "c"
indices[d] = "d"

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


def test_2x3_edge():
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
    z = 3 # end index, arrow to right, or arrow u

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

    # Set all bottom and left edge indices to a
    # row 0 
    TX0 = einsum("uI,UIR->uRU", T[0,0,:,a,a,:], T[0,1,:,:,a,:])
    TX0 = einsum("uIU,vIr->urvU", TX0, T[0,2,:,:,a,:])

    # row 1
    TX1 = einsum("udI,UIDR->udDRU", T[1,0,:,a,:,:], T[1,1,:,:,:,:])
    TX1 = einsum("udDIU,vIer->udDervU", TX1, T[1,2,:,:,:,:])

    # contract down
    TSQ = einsum("uIJKrvU,IRKJ->uRrvU", TX1, TX0)
    TSQ.dump("tsq_file")

    term1= TSQ[a,z,a,a,a] # 1st row
    term2= TSQ[a,a,z,a,a] # 2nd row
    term3= TSQ[z,a,a,a,a] # 1st col
    term4= TSQ[a,a,a,a,z] # 2nd col
    term5= TSQ[a,a,a,z,a] # 3rd col
    
    # c*x01*x10 + c*x02*(c*x10 + x11)
    # + x00*x01 + x00*x10 + x02*(c*x00 + x01)
    # + x10*x11 + x11*(c*x00 + x01)
    # + x12*(c*x10 + x11) + x12*(c*(c*x00 + x01) + x02)
    print (term1 + term2 + term3 + term4 + term5)

def get_top_mps(D, L):
    """
    Top MPS,with dangling bond to right

    <---- L ---->  
    ________..._ 
    | | |       | <- dangling bond _
    The top L physical indices are
    stored as a single vector index.
    The dangling bond is of the same dimension
    as the physical index. 

    D is the bond dimension of the PEPO
    """
    #mps = np.zeros([D]*(L+1),np.object) # L+1 because of dangling bond
    mps = np.zeros([D]*(L+1)) # L+1 because of dangling bond

    #  phys dangler
    # (0000) 1      = 1
    # (1000) 0      = 1
    # (0100) 0      = 1
    #  ... ... ...

    for i in range(L+1):
        index = [a]*(L+1)
        index[i] = z
        mps[tuple(index)] = 1

    return np.reshape(mps, [np.prod(mps.shape)])

def get_right_mps(D, L):
    """
    Returns right boundary mps

    __|
    __|
    __|
    
    """
    mps = np.zeros([L], np.object)

    #ten0 = np.zeros([D, D],np.object) # <-- bottom-most
    #teni = np.zeros([D, D, D],np.object) # <-- general
    ten0 = np.zeros([D, D]) # <-- bottom-most
    teni = np.zeros([D, D, D]) # <-- general

    ten0[z, z] = 1
    ten0[a, a] = 1

    teni[z,a,z] = 1
    teni[a,a,a] = 1
    teni[z,z,a] = 1
    
    mps[0] = ten0
    for i in range(1, L):
        mps[i] = teni
    return mps
    
def contract_pepo_pbc():
    """
    Contracts a periodic PEPO with no boundary
    """
    pass

def contract_pepo_obc(pepo, mps_top, mps_right):
    """
    Contracts PEPO with MPS top boundary
    and MPS right boundary

     mps top
    ------..-- 
    | | |    |    
    :   :    :
    PEPO:    :
    |_|_|_.._| 2
    |_|_|_.._| 1  <- mps right
    |_|_|_.._| 0
    """
    
    nr,nc = pepo.shape

    # turn bottom row => MPS vector
    # left and down must be "a" inputs
    mps0 = pepo[0,0][:,a,a,:]
    for i in range(1,nc):
        mps0 = einsum("uI,UIr->uUr", mps0, pepo[0,i][:,:,a,:])

        mps0 = np.reshape(mps0, [mps0.shape[0]*mps0.shape[1], mps0.shape[2]])

    # contract with bottom tensor of mps_right
    mps0 = einsum("uI,UI->uU", mps0, mps_right[0])
    mps0 = np.reshape(mps0, [mps0.shape[0]*mps0.shape[1]])

    # turn other rows into MPO mat, and contract into MPS vector
    # leftmost must be "a" input
    for i in range(1,nr):
        mpo = pepo[i,0][:,a,:,:]
        for j in range(1,nc):
            mpo = einsum("udI,UIDr->uUdDr", mpo, pepo[i,j])
            mpo = np.reshape(mpo, [mpo.shape[0]*mpo.shape[1], mpo.shape[2]*mpo.shape[3], mpo.shape[4]])

        # contract with tensor of mps_right
        mpo = einsum("udI,UID->uUdD", mpo, mps_right[i])
        mpo = np.reshape(mpo, [mpo.shape[0]*mpo.shape[1],mpo.shape[2]*mpo.shape[3]])

        # mul into mps
        mps0 = np.dot(mpo, mps0)

    # contract with mps_top boundary
    # This is a special state which ensures that only a single
    # outgoing z index is summed over
    scalar = np.dot(mps_top, mps0)

    return scalar
                       

def test_PEPO():

    # size of lattice
    pepo = np.zeros([3,3], dtype=object)

    # Tensors have input and output arrow, left->right, down->up
    for i in range(pepo.shape[0]):
        for j in range(pepo.shape[1]):

            T = np.zeros([4,4,4,4], dtype=object)
            T[a,a,a,a] = 1
            T[a,z,a,z] = 1
            T[z,a,z,a] = 1
            
            T[a,a,a,b] = sym.symbols("x"+str(i)+str(j)) # 1 
            T[a,b,a,z] = sym.symbols("x"+str(i)+str(j)) # 1
            T[c,a,a,a] = sym.symbols("x"+str(i)+str(j)) # 1
            T[b,a,a,a] = sym.symbols("x"+str(i)+str(j)) #
            
            T[z,a,b,a] = sym.symbols("x"+str(i)+str(j)) # 2

            T[b,b,a,a] = 1#sym.symbols("c") # 2 
            T[a,b,a,b] = 1#sym.symbols("c") # 2 
            T[b,a,b,a] = 1#sym.symbols("c") # 2 
            T[c,a,c,a] = 1#sym.symbols("c")
            
            T[z,b,c,a] = 1#sym.symbols("c") # 3 
            T[a,b,a,z] = sym.symbols("x"+str(i)+str(j)) # 4

            pepo[i,j] = T

    mps_top = get_top_mps(4, pepo.shape[1])
    mps_right = get_right_mps(4, pepo.shape[0])

    scalar = contract_pepo_obc(pepo, mps_top, mps_right)
    
    coeffs = scalar.expand().as_coefficients_dict()

    print "PEPO terms"
    for key in coeffs:
        nx = str(key).count("x")
        if nx == 1:
            raise RuntimeError
        if nx == 2:
            print coeffs[key], key

    print "Total number of distinct terms", len(coeffs.keys())
    
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
    
    # c*x01*x10 + c*x02*(c*x10 + x11)
    # + x00*x01 + x00*x10 + x02*(c*x00 + x01)
    # + x10*x11 + x11*(c*x00 + x01)
    # + x12*(c*x10 + x11) + x12*(c*(c*x00 + x01) + x02)

    
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
    
