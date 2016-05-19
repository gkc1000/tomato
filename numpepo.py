import math
import numpy as np
import scipy as sp
import scipy.linalg

I = np.eye(2)
Sz = 2*np.array([[.5, 0.], [0., -0.5]])

def modelN(n,eps,h=0):
   T,bra = get_mpo_nn(eps,h)
   Tmp = T.copy()
   for i in range(1,n):
      Tmp = np.einsum('lrud,rRUD->lRuUdD',Tmp,T)
      s = Tmp.shape
      Tmp = Tmp.reshape((s[0],s[1],s[2]*s[3],s[4]*s[5]))
   return Tmp,bra

def get_mpo_nn(eps,h=0):
    T = np.zeros([2,2,2,2])
    T[0,0,:,:] = I
    
    aeps = math.sqrt(abs(eps))
    if eps>0.:
       sgn = 1.0
    else:
       sgn = -1.0
    T[0,1,:,:] = aeps * Sz
    T[1,0,:,:] = sgn*aeps * Sz
    
    if abs(h) > 1.e-12: T[0,0] += eps*h*Sz
    #T[0,1,:,:] = Sz
    #T[1,0,:,:] = eps*Sz

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
        tmp = T.reshape(s[0],s[1]*s[2]*s[3])
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

def time_evol():
    ns = 11
    eps = 0.1 #-0.1 #-0.01/2**ns

    #T0 = heisenT(eps)
    #Z0 = np.dot(np.dot(bra, np.einsum("lrNN->lr", T)), bra)
    
    T, bra = get_mpo_nn(eps)
    nsite = 2
    T,bra = modelN(nsite,eps,h=0.)

    #
    # (1-eH)^N, ||H||^n get large very quickly.
    #
    D = 100
    logRenorm = 0.
    scale = True #False
    for i in range(2):#+ns):
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

    #print trT
    xsite = 4/nsite
    Tn = np.linalg.matrix_power(trT,xsite)
    Z = np.dot(np.dot(bra, Tn), bra)
    print 
    print 'Z=',Z
    sumlnZ = math.log(Z)+xsite*logRenorm
    nsite *= xsite
    print 'nsite=',nsite
    print 'sum=',sumlnZ,sumlnZ/nsite


def boundary_check():

   #M = heisenT(0.)

   # different values of beta
   # we are multiplying out (1-eps H)(1-eps H)
   #
   #
   print "eps  E (from tr)  E (from bra/ket)"

   for eps in [0, 0.005, 0.01, 0.02]:

        #M = heisenT(math.sqrt(2)*eps)
        T, bra = get_mpo_nn(eps) # boundary is vacuum state

        D = 4
        TT, bra = contract_down(T, bra, 4)
        M = np.einsum("lrII->lr", TT)

        E_fac = 0.

        # a quick demonstration using the 
        # Ising tensor, that the free energy is invariant to boundary

        # do 2**20 sites
        for i in range(20):
            M = np.dot(M,M)
            fac = np.linalg.norm(M)
            M /= fac

            E_fac *= 2
            E_fac += math.log(fac)
            #print E_fac

            nsites = 2**(i+1)

	    bound_E = (math.log(np.dot(np.dot(bra, M), bra.T)) + E_fac) / nsites
            tr_E = (math.log(np.trace(M)) + E_fac)/nsites
            print i,nsites*bound_E,nsites*tr_E
            
            #print "# sites %i: free energy (tr) %10.6f, (bound) %10.6f, (bound_rand) %10.6f" % (nsites, tr_E / nsites, bound_E / nsites,
            #                                                                                            bound_rand_E / nsites)
        print eps, bound_E-math.log(2), tr_E-math.log(2) # looks like eps**2


        
# def get_smpo_nn(eps):
#     T = np.zeros([3,3,2,2])
#     T[0,0,:,:] = I
# 
#     aeps = math.sqrt(abs(eps))
#     if eps>0.:
#        sgn = 1.0
#     else:
#        sgn = -1.0
#     T[0,1,:,:] = aeps * Sz
#     T[1,2,:,:] = sgn*aeps * Sz
#     T[2,2,:,:] = I
# 
#     return T
# 
# #eps=0.1
# #T = get_smpo_nn(eps)
# #print 'xx',np.einsum('abii',T)
# #print 'xx',np.einsum('abij,cdjk->ab',T,T)

time_evol()
#boundary_check()
