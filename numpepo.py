import math
import numpy as np
import scipy as sp
import scipy.linalg

I = np.eye(2)
Sz = np.array([[.5, 0.], [0., -0.5]])

def modelN(n,eps):
   T,bra = get_mpo_nn(eps)
   Tmp = T.copy()
   for i in range(1,n):
      Tmp = np.einsum('lrud,rRUD->lRuUdD',Tmp,T)
      s = Tmp.shape
      Tmp = Tmp.reshape((s[0],s[1],s[2]*s[3],s[4]*s[5]))
   return Tmp,bra

def get_mpo_nn(eps):
    T = np.zeros([2,2,2,2])
    T[0,0,:,:] = I
    
    aeps = math.sqrt(abs(eps))
    if eps>0.:
       sgn = 1.0
    else:
       sgn = -1.0
    T[0,1,:,:] = aeps * Sz
    T[1,0,:,:] = sgn*aeps * Sz
    
    #T[0,1,:,:] = Sz
    #T[1,0,:,:] = eps*Sz

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

    #print "truncations", ltrunc, rtrunc
    
    # TT : lrud
    TT = np.einsum("lrud,la->arud", TT, Ulr)
    TT = np.einsum("lrud,ra->laud", TT, Ulr)

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
    eps = 0.01

    #T0 = heisenT(eps)
    #Z0 = np.dot(np.dot(bra, np.einsum("lrNN->lr", T)), bra)
    
    T, bra = get_mpo_nn(eps)
    #T,bra = modelN(2,eps)

    #
    # (1-eH)^N, ||H||^n get large very quickly.
    #
    D = 4
    logRenorm = 0.
    scale = True #False
    for i in range(20):
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
        Z = np.dot(np.dot(bra, np.einsum("lrNN->lr", T)), bra)
	print
	print 'iter=',i
	if scale:
	   print 'Z=',Z,math.log(Z)
	   sumlnZ = (math.log(Z)+logRenorm)
	   print 'sum=',sumlnZ,math.exp(sumlnZ)
	else:
	   print 'Z=',Z,math.log(Z)

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
