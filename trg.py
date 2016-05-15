import math
import numpy as np
import numpy.random

import scipy as sp
import scipy.linalg

def hosvd4(T):
    DMa=np.einsum("ajkl,Ajkl->aA",T,T)
    DMb=np.einsum("ibkl,iBkl->bB",T,T)
    DMc=np.einsum("ijcl,ijCl->cC",T,T)
    DMd=np.einsum("ijkd,ijkD->dD",T,T)

    eiga,veca=sp.linalg.eigh(DMa)
    eigb,vecb=sp.linalg.eigh(DMb)
    eigc,vecc=sp.linalg.eigh(DMc)
    eigd,vecd=sp.linalg.eigh(DMd)

    Sa=np.einsum("ijkl,ia->ajkl",T,veca)
    Sb=np.einsum("ajkl,jb->abkl",Sa,vecb)
    Sc=np.einsum("abkl,kc->abcl",Sb,vecc)
    S=np.einsum("abcl,ld->abcd",Sc,vecd)

    return S,veca,vecb,vecc,vecd,eiga,eigb,eigc,eigd

def contract_down(T, D):
    M=np.einsum("uilr,iDLR->uDlLrR",T,T)
    # u,d,l,r reshaping and HOSVD
    M=np.reshape(M,[M.shape[0],M.shape[1],M.shape[2]*M.shape[3],M.shape[4]*M.shape[5]])
    S,Uu,Ud,Ul,Ur,eigu,eigd,eigl,eigr=hosvd4(M)

    # make truncated lr basis
    bigD=len(eigl)
    ltrunc=np.sum(eigl[bigD-D:bigD])
    rtrunc=np.sum(eigr[bigD-D:bigD])

    if D<bigD and ltrunc<rtrunc:
        Ulr=Ul[:,bigD-D:bigD]
    else:
        Ulr=Ur[:,bigD-D:bigD]

    # transform to truncated lr basis
    Ma=np.einsum("udlr,la->udar",M,Ulr)
    Mlr=np.einsum("udar,rb->udab",Ma,Ulr)

    print "trunc", norm2D(Mlr),norm2D(M)
    return Mlr
    
def contract_right(T, D):
    M=np.einsum("udli,UDir->uUdDlr",T,T)

    # y,z,l,r reshaping and HOSVD
    M=np.reshape(M,[M.shape[0]*M.shape[1],M.shape[2]*M.shape[3],M.shape[4],M.shape[5]])
    S,Uu,Ud,Ul,Ur,eigu,eigd,eigl,eigr=hosvd4(M)

    # make truncated ud basis
    bigD=len(eigu)
    utrunc=np.sum(eigu[bigD-D:bigD])
    dtrunc=np.sum(eigd[bigD-D:bigD])

    if D<bigD and utrunc<dtrunc:
        Uud=Uu[:,bigD-D:bigD]
    else:
        Uud=Ud[:,bigD-D:bigD]

    # transform to truncated ud basis
    Ma=np.einsum("udlr,ua->adlr",M,Uud)
    Mud=np.einsum("adlr,db->ablr",Ma,Uud)

    print "trunc", norm2D(Mud),norm2D(M)

    return Mud

def trace2D(T):
    T0=np.einsum("uulr",T)
    return np.einsum("ll",T0)

def norm2D(T):
    return math.sqrt(trace2D(np.abs(T)))

def contract2D(T, D, niter, scale=True):
    # contract n iterations
    Tn=T
    logRenorm=0.

    for i in xrange(niter):
        print "Iteration", i
        if scale:
            logRenorm*=4.
            Tnorm=norm2D(Tn)
            logRenorm+=2*math.log(Tnorm)
            Tn/=Tnorm

        T1=contract_down(Tn,D)
        # 2 sets of down renormalizations are done for
        # each set of right renormalizations, 
        # so first Tn norm appears again
        if scale:
            logRenorm+=2*math.log(Tnorm)
            Tnorm=norm2D(T1)
            T1/=Tnorm
            logRenorm+=2*math.log(Tnorm)

        Tn=contract_right(T1,D)

    # apply pbc to final tensor
    return trace2D(Tn),logRenorm

def heisenT(beta):
    # Following notation of Xiang (http://arxiv.org/pdf/1201.1144v4.pdf)
    W=np.array([[math.sqrt(math.cosh(beta)), math.sqrt(math.sinh(beta))],
               [math.sqrt(math.cosh(beta)), -math.sqrt(math.sinh(beta))]])

    T=np.einsum("au,ad,al,ar->udlr",W,W,W,W)
    return T

def test():
    np.random.seed(2)
    beta=0.1
    T=heisenT(beta)


    niter=10
    D=4
    Z,logZnorm=contract2D(T,D,niter)
    print "final---------"
    print math.log(Z),logZnorm
    F=(math.log(Z)+logZnorm)/(4**niter)
    print "T, energy per site", 1./beta, -1./beta*F
