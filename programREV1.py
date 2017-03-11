import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros as np, amax
from pylab import *
from numpy import *
import scipy.sparse as sparse
import scipy.linalg as linalg
import numpy.linalg as LA
from pandas.tools.plotting import scatter_plot
from matplotlib.pyplot import scatter

def load_mnist(dataset="training", digits=range(10), path=r"C:\Users\Robert Shultz\Google Drive\Machine Learning\Assignment 3\ImageData"):
    """
    Adapted from: http://cvxopt.org/applications/svm/index.html?highlight=mnist
    """
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx')
        fname_lbl = os.path.join(path, 't10k-labels.idx')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")
    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()
    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()
    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)
    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]
    return images, labels

images, labels = load_mnist('training', digits=[2,3])
flatimages = list()
for i in images:
    flatimages.append(i.ravel())
X = np.asarray(flatimages)
T = labels

"""X is the trainging data"""
"""T is the class labels"""

# """This is a test of the matplotlib print"""
# plt.imshow(X[1].reshape(28, 28),interpolation='None', cmap=cm.gray )
# show()


def xzcvpr(X):
    u=np.mean(X,axis=0)
    Z=X-u
    C=np.cov(Z,rowvar=False)
    [L,V]=LA.eigh(C)
    L=np.flipud(L)
    V=np.flipud(V.T)  
    P = np.dot(Z,V.T)
    P=P[:,:2]
    P = np.append(P,np.zeros([len(P),1]),1)
#     P[:,2]=T[:,0]
    print((P[:6,:]))
    print((T[:6,:]))
    P[:,2]=P[:,2]+T[:,0]
    print((P[:6,:])) 
    Pn = P[(P[:,2]==2)]
    Pp = P[(P[:,2]==3)]
    print(Pn[:5,:])
    print(Pp[:5,:])
    plt.scatter(Pn[:,0],Pn[:,1],color='red', s = 1)
    plt.scatter(Pp[:,0],Pp[:,1],color='blue', s = 1)
    plt.show()
    
#     Pn = P[P[:,2]==2]
#     print(P[4,:])
#     print(unique(labels))
#     print(Pn[50,:])
#     R = np.dot(P,V)
#     scatter(P[:,0],P[:,1], s=5)
#     plt.show()


xzcvpr(X)




# #     [print(a) for a in P[:,:]]
#     hmin = amin(P[:,0])
#     hmax = amax(P[:,0])
#     vmin = amin(P[:,1])
#     vmax = amax(P[:,1])


