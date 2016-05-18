import math
import numpy as np
import scipy as sp
import scipy.linalg

I = np.eye(2)
Sz = np.array([[.5, 0.], [0., -0.5]])

def get_mpo_nn(eps):
    T = np.zeros([2,2,2,2])
    T[0,0,:,:] = I
    T[0,1,:,:] = Sz
    T[1,0,:,:] = eps * Sz

    bra = np.zeros([2])
    bra[0] = 1.

    return T, bra

def hosvd(T,index_type):
    if index_type == "l":
        DM=np.einsum("ajkl,Ajkl->aA",T,T)
    elif index_type == "r":
        DM=np.einsum("ibkl,iBkl->bB",T,T)
    elif index_type == "u":
        DM=np.einsum("ijcl,ijCl->cC",T,T)
    elif index_type == "d":
        DM=np.einsum("ijkd,ijkD->dD",T,T)
    else:
        raise RuntimeError

    print "DM"
    print DM

    eig,vec=sp.linalg.eigh(DM)
    return eig, vec

def contract_down(T, bra, D):
    TT = np.einsum("lruI,LRId->lLrRud", T, T)

    bb = np.einsum("l,L->lL", bra, bra)
    bb = np.reshape(bb, [bb.shape[0]*bb.shape[1]])
    
    TT = np.reshape(TT, [TT.shape[0]*TT.shape[1],
                     TT.shape[2]*TT.shape[3],
                     TT.shape[4],
                     TT.shape[5]])
                     
    eigl, vecl = hosvd(TT,"l")
    eigr, vecr = hosvd(TT,"r")

    # make truncated lr basis
    bigD=len(eigl)
    ltrunc=np.sum(eigl[bigD-D:bigD])
    rtrunc=np.sum(eigr[bigD-D:bigD])

    # choose either the left vectors or right vectors
    # depending on which gives a smaller truncation
    if D<bigD and ltrunc<rtrunc:
        Ulr=vecl[:,bigD-D:bigD]
    else:
        Ulr=vecr[:,bigD-D:bigD]

    print "truncations", ltrunc, rtrunc
    
    # TT : lrud
    TT = np.einsum("lrud,la->arud", TT, Ulr)
    TT = np.einsum("lrud,ra->laud", TT, Ulr)

    bb = np.einsum("l,la->a", bb, Ulr)

    return TT, bb

def heisenT(beta):
    # Following notation of Xiang (http://arxiv.org/pdf/1201.1144v4.pdf)
    W=np.array([[math.sqrt(math.cosh(beta)), math.sqrt(math.sinh(beta))],
               [math.sqrt(math.cosh(beta)), -math.sqrt(math.sinh(beta))]])

    T=np.einsum("au,ad,al,ar->udlr",W,W,W,W)
    return T


def time_evol():
    eps = +0.0

    T0 = heisenT(eps)
    #Z0 = np.dot(np.dot(bra, np.einsum("lrNN->lr", T)), bra)
    
    T, bra = get_mpo_nn(eps)
    print T
    D = 4
    for i in range(1):
        T, bra = contract_down(T, bra, D)
        Z = np.dot(np.dot(bra, np.einsum("lrNN->lr", T)), bra)
        print Z
    
    

