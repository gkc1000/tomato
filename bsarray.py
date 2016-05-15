import numpy as np
import bsarray_helper
import null

NULL = null.NULL
dot = bsarray_helper.dot
einsum = bsarray_helper.einsum

class BSArray(np.ndarray):
    """
    Block sparse array: forwards operations onto
    elements of the array
    """
    def __array_finalize(self, obj):
        if obj is None and obj.dtype != object:
            raise RuntimeError

    def reshape(self, shape, order="C"):
        """
        reshapes array, and reshapes each element.
        If element is NULL, then it will be skipped
        by the try/except clause
        """
        self = np.ndarray.reshape(self, shape, order=order)
        for ix in np.ndindex(self.shape):
            try:
                self[ix] = self[ix].reshape(shape, order=order)
            except:
                pass
        return self
    
    def transpose(self, axes=None):
        """
        reshapes array, and reshapes each element.
        If element is NULL, then it will be skipped
        by the try/except clause
        """
        self = np.ndarray.transpose(self, axes)
        for ix in np.ndindex(self.shape):
            try:
                self[ix] = self[ix].transpose(axes)
            except:
                pass
        return self
    
    def dot(self, other):
        return bsarray_helper.dot(self, other)
    
def consistent_shape(a):
    """
    Check that along a given index
    all blocks have the same dimension, e.g.

    a[0,2] and a[1,2] -- a[0,2].shape[1] == a[1,2].shape[1]
    a[1,1] and a[1,3] -- a[1,1].shape[0] == a[1,3].shape[0]

    and return those dimensions
    """
    pass
    
def test():
    try:
        a = BSArray(np.empty([2,2]))
    except:
        print "Need object as dtype"

    a = np.empty([2,2], dtype=object)
    print a
    
    a = a.view(BSArray)

    a = np.array([NULL]*4).view(BSArray)
    a = np.ndarray.reshape(a, [2,2], order="C")
    a = a.reshape([2,2], order="C")
    print a.shape
    print a.__class__.__name__

    a2 = a
    a2 = np.reshape(a2, [2,2])
    print a2.__class__.__name__

    print "-----"
    b = a
    
    print a
    
    b = a
    c = a.dot(b)

    print "cello\n", c
    d = einsum("ij,jk->ik", a, b)

    print "dello\n", d

    print "c-d\n"
    print c - d
    
    a[0,0] = np.eye(2)*2
    b[0,0] = np.ones([2,2])
    
    c = a.dot(b)
    #c = dot(a, b)
    d = einsum("ij,jk->ik", a, b)

    print "C\n", c
    print "D\n", d
    print d.dtype
    print d.__class__.__name__
    
    e = c+d
    print "E\n", e
    print e.dtype
    print e.__class__.__name__

    f = e*2

    print "F\n", f
    
