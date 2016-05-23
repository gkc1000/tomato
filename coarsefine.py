#
# Results:
# 
# c1/c2/c4/c10= 0.367879441171 0.606530659713 0.778800783071 0.904837418036
# val1 = 0.606983721063
# val2 = 0.529103148948
# val2b= 0.211612363815
# val2c= 0.470656563107
# T1.shape = (10, 10, 2, 2)
# T2.shape = (20, 20, 64, 64)
# T2b.shape= (10, 10, 64, 64)
# T2c.shape= (10, 10, 64, 64)
# 

from scipy.special import binom
import numpy as np
import math
import numpepo
import itertools

###########################################
# 1D interaction: V(x) = exp(-lambda*|x|)
###########################################

def ctensor(n,nmax=1):
# 
# [1, 1, 0, 0, 0]
# [2, 0, 1, 0, 0]
# [3, 0, 0, 1, 0]
# [4, 0, 0, 0, 1]
# [5, 1, 1, 0, 0]
# [6, 1, 0, 1, 0]
# [7, 1, 0, 0, 1]
# [8, 0, 1, 1, 0]
# [9, 0, 1, 0, 1]
# [10, 0, 0, 1, 1]
# 
   dims = [int(binom(n,k)) for k in range(nmax+1)]
   ndim = sum(dims)
   t = np.zeros([ndim]+[2]*(n))
   t[tuple([0]*(n+1))] = 1.0
   ioff = 0
   for nm in range(1,nmax+1):
      for ic in itertools.combinations(range(n),nm):
	 lst = [0]*n
	 for j in ic:
	    lst[j] = 1
	 ioff += 1
	 idx = [ioff]+lst
         t[tuple(idx)] = 1.0
   t = t.reshape(ndim,2**n)
   return t

def test():
    res = [0]*3
    nlayer = 1 
    
    ns = 5
    eps = -0.01/4/2 #nlayer #0.1 #-0.01
    beta = abs(eps)*2**ns
    print 'eps =',eps
    print 'beta=',beta

    lambda0 = 1.0
    c1 = math.exp(-lambda0) # h=1.0
    c2 = math.exp(-lambda0/2.0) # h=0.5
    c4 = math.exp(-lambda0/4.0) # 
    c6 = math.exp(-lambda0/6.0) # 
    print 'c1/c2/c4/c6=',c1,c2,c4,c6
    # c1/c2/c4/c6= 0.367879441171 0.606530659713 0.778800783071 0.846481724891

    D = 30
    nclst = 1
    c = c1
    T,bra = numpepo.get_mpo_exp(eps,c,h=0.,iop=0)
    T = numpepo.modelN(T,nclst)
    T,bra = numpepo.modelM(T,bra,nlayer)
    val1,T1 = evol(T,bra,ns,D,nclst)
    print 'val1=',val1

    D = 30
    nclst0 = 6
    c = c6
    T,bra = numpepo.get_mpo_exp(eps,c,h=0.,iop=0)
    T = numpepo.modelN(T,nclst0)
    T,bra = numpepo.modelM(T,bra,nlayer)
    # Map
    ct = ctensor(nclst0,nmax=1)
    ct0 = np.zeros((2,ct.shape[1]))
    ct0[0] = ct[0]
    icase = 0 #1
    if icase == 0:
       for iclst in range(1,nclst0+1):
          ct0[1] += ct[iclst]
       ct0[1] = ct0[1]/math.sqrt(nclst0)
    else:
       ct0[1] = ct[1]
    T = np.einsum('uU,lrUD,dD->lrud',ct0,T,ct0)
    val2,T2 = evol(T,bra,ns,D,nclst0)
    print 'val2=',val2

    D = 30
    nclst = 1
    c = c2
    T,bra = numpepo.get_mpo_exp(eps,c,h=0.,iop=0)
    T = numpepo.modelN(T,nclst)
    T,bra = numpepo.modelM(T,bra,nlayer)
    val2b,T2b = evol(T,bra,ns,D,nclst)
    print 'val2b=',val2b

    D = 10
    nclst = 2
    c = c2
    T,bra = numpepo.get_mpo_exp(eps,c,h=0.,iop=0)
    T = numpepo.modelN(T,nclst)
    T,bra = numpepo.modelM(T,bra,nlayer)
    val2c,T2c = evolNEW(T,bra,ns,D,nclst,T1)
    print 'val2c=',val2c

    print 
    print 'c1/c2/c4/c6=',c1,c2,c4,c6
    print 'val1 =',val1
    print 'T1.shape =',T1.shape
    print 'val2 =',val2*nclst0
    print 'T2.shape =',T2.shape
    print 'val2b=',val2b
    print 'T2b.shape=',T2b.shape
    print 'val2c=',val2c
    print 'T2c.shape=',T2c.shape
    return 0


def evol(T,bra,ns,D,nclst):
    T,bra,trT,logRenorm = numpepo.time_evol(T,bra,ns,D)
    xsite = 20 
    sumlnZ = numpepo.contract_x(xsite,trT,logRenorm)
    nsite = 2**xsite
    val = sumlnZ/(nsite*nclst)
    print 'nsite=',nsite
    print 'sum=',sumlnZ,val
    return val,T


# TEST
def evolNEW(T,bra,ns,D,nclst,T0):
    T,bra,trT,logRenorm = time_evol(T,bra,ns,D,nclst,T0)
    xsite = 20 
    sumlnZ = numpepo.contract_x(xsite,trT,logRenorm)
    nsite = 2**xsite
    val = sumlnZ/(nsite*nclst)
    print 'nsite=',nsite
    print 'sum=',sumlnZ,val
    return val,T

def time_evol(T,bra,ns,D,nclst,T0):
    #
    # (1-eH)^N, ||H||^n get large very quickly.
    #
    logRenorm = 0.
    scale = True #False
    for i in range(ns):
	Tnorm = np.linalg.norm(T)
	Bnorm = np.linalg.norm(bra)
	print 'Tnorm=',Tnorm,'Bnorm=',Bnorm
	if scale:
	   logRenorm *= 2
	   logRenorm += 2*math.log(Tnorm)
	   T = T/Tnorm
	   logRenorm += 4*math.log(Bnorm)
	   bra = bra/Bnorm
	T, bra = contract_down(T, bra, D, nclst, T0)
	#T = contract_left(T,30)
 	trT = np.einsum("lrNN->lr", T)
        Z = np.dot(np.dot(bra, trT), bra)
	print
	print 'iter=',i
	if scale:
	   print 'Z=',Z,math.log(Z)
	   sumlnZ = (math.log(Z)+logRenorm)
	   print 'sum=',sumlnZ
	else:
	   print 'Z=',Z,math.log(Z)
    return T,bra,trT,logRenorm

def contract_down(T, bra, D, nclst, T0):
    TT = np.tensordot(T,T,axes=([3],[2])) # lruLRd
    TT = TT.transpose(0,3,1,4,2,5)
    TT = np.reshape(TT, [TT.shape[0]*TT.shape[1],
                     TT.shape[2]*TT.shape[3],
                     TT.shape[4],
                     TT.shape[5]])
    #=============================================
    #      |
    #     ===
    #      |
    #     / \
    #    -----
    #     | |
    #   =======
    #     | |
    ct = ctensor(nclst,nmax=1)
    # (5, 16) => [2,16]
    ct0 = np.zeros((2,ct.shape[1]))
    ct0[0] = ct[0]
    for iclst in range(1,nclst+1):
       ct0[1] += ct[iclst]
    ct0[1] = ct0[1]/math.sqrt(nclst)

#    # (4, 4, 2, 2) => (4,4,2,16)
#    T0c = np.einsum('lrud,dD->lruD',T0,ct0)
#    # (2,16)
#    T0x = np.einsum('lluD->uD',T0c)
#    TT0 = np.einsum('uU,lrUD,dD->lrud',T0x,TT,T0x)

    TT0 = np.einsum('uU,lrUD,dD->lrud',ct0,TT,ct0)

    TTnorm = np.linalg.norm(TT0)
    TT0 = TT0/TTnorm
    print 'TTnorm=',TTnorm
    eigl, vecl = numpepo.hosvd(TT0,"l")
    eigr, vecr = numpepo.hosvd(TT0,"r")
    #=============================================   
    # make truncated lr basis
    bigD=len(eigl)
    Deff=min(D,bigD)
    ltrunc=np.sum(eigl[:bigD-Deff])
    rtrunc=np.sum(eigr[:bigD-Deff])
    # choose either the left vectors or right vectors
    # depending on which gives a smaller truncation
    if Deff<bigD and ltrunc<rtrunc:
        Ulr=vecl[:,bigD-Deff:bigD]
    else:
        Ulr=vecr[:,bigD-Deff:bigD]
    print 'bigD/ltrunc/rtrunc=',bigD,ltrunc,rtrunc,bigD-Deff,Ulr.shape
    print '     lsigs /rsigs =',np.sum(eigl),np.sum(eigr)
    # TT : lrud
    #TTx = np.einsum("lrud,la->arud", TT, Ulr)
    #TTx = np.einsum("lrud,ra->laud", TTx, Ulr)
    TT = np.tensordot(TT,Ulr,axes=([0],[0])) # rudl
    TT = np.tensordot(TT,Ulr,axes=([0],[0])) # udlr
    TT = TT.transpose(2,3,0,1)
    bb = np.einsum("l,L->lL", bra, bra)
    bb = np.reshape(bb, [bb.shape[0]*bb.shape[1]])
    bb = np.einsum("l,la->a", bb, Ulr)
    return TT, bb


if __name__ == '__main__':
   #t = ctensor(4)
   #print 'proj=',np.einsum('ijklt,mjklt->im',t,t)
   #print 'norm=',np.einsum('ijklt,ijklt',t,t)
   
   #n = 4
   #nmax = 4
   #dims = [int(binom(n,k)) for k in range(nmax+1)]
   #ndim = sum(dims)
   #print ndim
   #
   #ctensor(n,nmax=2)
   
   test()
