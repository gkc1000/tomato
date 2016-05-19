import math
import numpy as np
import scipy as sp
import scipy.linalg

I = np.eye(2)
Sz = 2*np.array([[.5, 0.], [0., -0.5]])

def modelN(T,n):
   Tmp = T.copy()
   for i in range(1,n):
      Tmp = np.einsum('lrud,rRUD->lRuUdD',Tmp,T)
      s = Tmp.shape
      Tmp = Tmp.reshape((s[0],s[1],s[2]*s[3],s[4]*s[5]))
   return Tmp

def get_mpo_nn(eps,h=0,iop=0):
    T = np.zeros([2,2,2,2])
    T[0,0,:,:] = I
   
    if iop == 0:
       aeps = math.sqrt(abs(eps))
       if eps>0.:
          sgn = 1.0
       else:
          sgn = -1.0
       T[0,1,:,:] = aeps * Sz
       T[1,0,:,:] = sgn*aeps * Sz
   
    else:
       T[0,1,:,:] = Sz
       T[1,0,:,:] = eps*Sz

    if abs(h) > 1.e-12: T[0,0] += eps*h*Sz

    bra = np.zeros([2])
    bra[0] = 1.

    return T, bra

def hosvd(T,index_type):
    s = T.shape
    if index_type == "l":
        tmp = T.reshape(s[0],s[1]*s[2]*s[3])
	DM = tmp.dot(tmp.T)
	#DM=np.einsum("ajkl,Ajkl->aA",T,T)
    elif index_type == "r":
        tmp = T.transpose(1,0,2,3)
        tmp = tmp.reshape(s[0],s[1]*s[2]*s[3])
	DM = tmp.dot(tmp.T)
	#DM=np.einsum("ibkl,iBkl->bB",T,T)
    elif index_type == "u":
        DM=np.einsum("ijcl,ijCl->cC",T,T)
    elif index_type == "d":
        DM=np.einsum("ijkd,ijkD->dD",T,T)
    else:
        raise RuntimeError
    eig,vec=sp.linalg.eigh(DM)
    print 'eig=',eig
    return eig, vec

def contract_down(T, bra, D):
    #TT = np.einsum("lruI,LRId->lLrRud", T, T)
    TT = np.tensordot(T,T,axes=([3],[2])) # lruLRd
    TT = TT.transpose(0,3,1,4,2,5)

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
    
    # TT : lrud
    #TT = np.einsum("lrud,la->arud", TT, Ulr)
    #TT = np.einsum("lrud,ra->laud", TT, Ulr)
    TT = np.tensordot(TT,Ulr,axes=([0],[0])) # rudl
    TT = np.tensordot(TT,Ulr,axes=([0],[0])) # udlr
    TT = TT.transpose(2,3,0,1)

    bb = np.einsum("l,la->a", bb, Ulr)

    return TT, bb

def heisenT(beta):
    # Following notation of Xiang (http://arxiv.org/pdf/1201.1144v4.pdf)
    W=np.array([[math.sqrt(math.cosh(beta)), math.sqrt(math.sinh(beta))],
               [math.sqrt(math.cosh(beta)), -math.sqrt(math.sinh(beta))]])

    #T=np.einsum("au,ad,al,ar->udlr",W,W,W,W)
    T=np.einsum("al,ar->lr",W,W)
    return T

def time_evol(T,bra,ns,nsite,D):
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
	T, bra = contract_down(T, bra, D)
 	trT = np.einsum("lrNN->lr", T)
        Z = np.dot(np.dot(bra, trT), bra)
	print
	print 'iter=',i
	if scale:
	   print 'Z=',Z,math.log(Z)
	   sumlnZ = (math.log(Z)+logRenorm)
	   print 'sum=',sumlnZ,math.exp(sumlnZ),sumlnZ/nsite
	else:
	   print 'Z=',Z,math.log(Z)

    return T,bra,trT,logRenorm

def contract_x(xsite,trT,logRenorm):
    #Tn = np.linalg.matrix_power(trT,xsite)
    tmp = logRenorm
    Tn = trT
    for i in range(xsite):
       Tn = Tn.dot(Tn)
       fac = np.linalg.norm(Tn)
       Tn /= fac
       tmp *= 2
       tmp += math.log(fac)
    Z = np.einsum('ii',Tn)
    sumlnZ = math.log(Z)+tmp
    #Z = np.dot(np.dot(bra, Tn), bra)
    #print 
    #print 'Z=',Z
    #sumlnZ = math.log(Z)+xsite*logRenorm
    return sumlnZ

def test():
    nclst = 2
    tmp = 0
    ns = 4 #10 #+ tmp #10
    eps = -0.01/2**(2+tmp) #11
    D = 40 
    res = [0]*2
    nsite = 40
    for iop in [0,1]:
       print '='*20
       print 'iop=',iop
       print '='*20
       T,bra = get_mpo_nn(eps,h=0.,iop=iop)
       T = modelN(T,nclst)
       beta = abs(eps)*2**ns
       xsite = 10 
       T,bra,trT,logRenorm = time_evol(T,bra,ns,nsite,D)
       sumlnZ = contract_x(xsite,trT,logRenorm)
       nsite = 2**xsite
       val = sumlnZ/(nsite*nclst)
       print 'eps =',eps
       print 'beta=',beta
       print 'nsite=',nsite
       print 'sum=',sumlnZ,val
       res[iop] = val 
       trT = heisenT(beta)
       sumlnZ = contract_x(xsite,trT,0.)
       val = sumlnZ/nsite
       print
       print 'sum=',sumlnZ,val

    print
    print res

test()	
