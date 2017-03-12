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

def load_mnist(dataset="training", digits=range(10), path=r"C:\Users\Robert\Documents\GitHub\Assignment-3\ImageData"):
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

def Build2DHistograms(X,T,B, xmin, xmax):
    Hn = np.zeros([B,B]).astype('int32')
    Hp = np.zeros([B,B]).astype('int32')
    RC = clip(around(np.round(((B-1)*(X-xmin)/(xmax-xmin)))).astype('int32'),0,B-1)
    for i,rc in enumerate (RC):
        if T[i]==2:
            Hn[rc[0],rc[1]]+=1
        else:
            Hp[rc[0],rc[1]]+=1
    data = [Hn,Hp]
    return data

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
    P[:,2]=P[:,2]+T[:,0] 
    Pn = P[(P[:,2]==2)]
    Pp = P[(P[:,2]==3)]
    mup=np.mean(Pp,axis=0)[:2]
    mun=np.mean(Pn,axis=0)[:2]  
    P1min = amin(P[:,0])
    P1max = amax(P[:,0])
    P2min = amin(P[:,1])
    P2max = amax(P[:,1])  
    
    Pmax = amax(P[:,:2])
    Pmin = amin(P[:,:2])
       
    plt.scatter(Pn[:,0],Pn[:,1],color='red', s = 10)
    plt.scatter(Pp[:,0],Pp[:,1],color='blue', s = 10)
    xlabel("P1")
    ylabel("P2")
    plt.plot([mup[0],mup[0]],[P2min,P2max],linewidth=5)
    plt.plot([mun[0],mun[0]],[P2min,P2max],color='red',linewidth=5)
    plt.plot([P1min,P1max],[mup[1],mup[1]],color='blue',linewidth=5)
    plt.plot([P1min,P1max],[mun[1],mun[1]],color='red',linewidth=5)
    plt.show()
    cn = np.cov(Pn[:,:2].T, ddof = 1)
    cp = np.cov(Pp[:,:2].T, ddof = 1)
    B = np.round(np.log2(len(P))+1,0)
    print(B)
    print(Build2DHistograms(P[:,:2],P[:,2] , B, Pmin, Pmax)[0])
    print('\n',Build2DHistograms(P[:,:2],P[:,2] , B, Pmin, Pmax)[1])

xzcvpr(X)

#     hmin = amin(P[:,0])
#     hmax = amax(P[:,0])
#     vmin = amin(P[:,1])
#     vmax = amax(P[:,1])


