import numpy as np
import bsarray as bs
import re
#import numpy_helper
import null

DEBUG=False

def einsum(idx_str, *tensors):
    """Perform a more efficient einsum via reshaping to a matrix multiply.

    Current differences compared to np.einsum:
    This assumes that each repeated index is actually summed (i.e. no 'i,i->i')
    and appears only twice (i.e. no 'ij,ik,il->jkl'). The output indices must
    be explicitly specified (i.e. 'ij,j->i' and not 'ij,j').

    # TODO: generalize to handle these cases
    """

    if '->' not in idx_str:
        raise RuntimeError
        #return np.einsum(idx_str,*tensors)

    if idx_str.count(',') > 1:
        indices  = re.split(',|->',idx_str)
        indices_in = indices[:-1]
        idx_final = indices[-1]
        n_shared_max = 0
        for i in range(len(indices_in)):
            for j in range(i):
                tmp = list(set(indices_in[i]).intersection(indices_in[j]))
                n_shared_indices = len(tmp)
                if n_shared_indices > n_shared_max:
                    n_shared_max = n_shared_indices
                    shared_indices = tmp
                    [a,b] = [i,j]
        tensors = list(tensors)
        A, B = tensors[a], tensors[b]
        idxA, idxB = indices[a], indices[b]
        idx_out = list(idxA+idxB)
        idx_out = "".join([x for x in idx_out if x not in shared_indices])
        C = einsum(idxA+","+idxB+"->"+idx_out, A, B)
        indices_in.pop(a)
        indices_in.pop(b)
        indices_in.append(idx_out)
        tensors.pop(a)
        tensors.pop(b)
        tensors.append(C)
        return einsum(",".join(indices_in)+"->"+idx_final,*tensors)

    A, B = tensors
    # Split the strings into a list of idx char's
    idxA, idxBC = idx_str.split(',')
    idxB, idxC = idxBC.split('->')
    idxA, idxB, idxC = [list(x) for x in [idxA,idxB,idxC]]

    if DEBUG:
        print "*** Einsum for", idx_str
        print " idxA =", idxA
        print " idxB =", idxB
        print " idxC =", idxC

    # Get the range for each index and put it in a dictionary
    rangeA = dict()
    rangeB = dict()
    #rangeC = dict()
    for idx,rnge in zip(idxA,A.shape):
        rangeA[idx] = rnge
    for idx,rnge in zip(idxB,B.shape):
        rangeB[idx] = rnge
    #for idx,rnge in zip(idxC,C.shape):
    #    rangeC[idx] = rnge

    if DEBUG:
        print "rangeA =", rangeA
        print "rangeB =", rangeB

    # Find the shared indices being summed over
    shared_idxAB = list(set(idxA).intersection(idxB))
    #if len(shared_idxAB) == 0:
    #    return np.einsum(idx_str,A,B)
    idxAt = list(idxA)
    idxBt = list(idxB)
    inner_shape = 1
    insert_B_loc = 0
    for n in shared_idxAB:
        # Bring idx all the way to the right for A
        # and to the left (but preserve order) for B
        idxA_n = idxAt.index(n)
        idxAt.insert(len(idxAt)-1, idxAt.pop(idxA_n))

        idxB_n = idxBt.index(n)
        idxBt.insert(insert_B_loc, idxBt.pop(idxB_n))
        insert_B_loc += 1

        inner_shape *= rangeA[n]

    if DEBUG:
        print "shared_idxAB =", shared_idxAB
        print "inner_shape =", inner_shape

    # Transpose the tensors into the proper order and reshape into matrices
    new_orderA = list()
    for idx in idxAt:
        new_orderA.append(idxA.index(idx))
    new_orderB = list()
    for idx in idxBt:
        new_orderB.append(idxB.index(idx))

    if DEBUG:
        print "Transposing A as", new_orderA
        print "Transposing B as", new_orderB
        print "Reshaping A as (-1,", inner_shape, ")"
        print "Reshaping B as (", inner_shape, ",-1)"

    A = _cp(A)
    B = _cp(B)

    At = A.transpose(new_orderA).reshape(-1,inner_shape)
    Bt = B.transpose(new_orderB).reshape(inner_shape,-1)

    shapeCt = list()
    idxCt = list()
    for idx in idxAt:
        if idx in shared_idxAB:
            break
        shapeCt.append(rangeA[idx])
        idxCt.append(idx)
    for idx in idxBt:
        if idx in shared_idxAB:
            continue
        shapeCt.append(rangeB[idx])
        idxCt.append(idx)

    new_orderCt = list()
    for idx in idxC:
        new_orderCt.append(idxCt.index(idx))

    #return np.dot(At,Bt).reshape(shapeCt).transpose(new_orderCt)
    # print "einsum"
    # temp = dot(At,Bt)
    # print "temp\n", temp
    # temp2 = dot(At,Bt).reshape(shapeCt)
    # print "temp2\n", temp2
    # temp3 = dot(At,Bt).reshape(shapeCt).transpose(new_orderCt)
    # print "temp3\n", temp3
    
    return np.dot(At,Bt).reshape(shapeCt).transpose(new_orderCt)

def _cp(a):
    return np.array(a, copy=False, order='C', dtype=object)

# def dot(a, b, out=None):
#     """
#     Sparse block matrix multiplication

#     # TODO: generalize to beyond matrices
#     """
#     c = np.array([null.NULL] * (a.shape[0]*b.shape[1]), dtype=object).view(bs.BSArray)
#     c = np.reshape(c, [a.shape[0], b.shape[1]])
#     print c
#     print a
#     print b
#     for i in range(a.shape[0]):
#         for j in range(a.shape[1]):
#             for k in range(b.shape[0]):
#                 c[i,k] += elem_dot(a[i,j], b[j,k])
#     return c

# def elem_dot(a, b):
#     """
#     np.dot extended to handle null
#     """
#     if a is null.NULL:
#         return null.NULL
#     elif b is null.NULL:
#         return null.NULL
#     else:
#         return np.dot(a, b)
        
