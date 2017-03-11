import os, struct
import matplotlib as plt
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros as np
from pylab import *
from numpy import *
import scipy.sparse as sparse
import scipy.linalg as linalg
import numpy.linalg as LA

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
###############################################
images, labels = load_mnist('training', digits=[2,3])
""" converting from NX28X28 array into NX784 array"""
flatimages = list()
for i in images:
    flatimages.append(i.ravel())
X = np.asarray(flatimages)
"""X is the trainging data"""
"""labels is the class labels"""
print("Check shape of matrix", X.shape)
print("Check Mins and Max Values",np.amin(X),np.amax(X))
print("\nCheck training vector by plotting image \n")
##############################################
# """This is a test of the matplotlib print"""
# plt.imshow(X[1].reshape(28, 28),interpolation='None', cmap=cm.gray )
# show()
##############################################
"Calculate the mean (u) of X"
print('u')
u=np.mean(X,axis=0)
print(u)
print('shape: ', u.shape)
print('\n\n\n')
##############################################
'Plot u'
plt.imshow(u.reshape(28, 28),interpolation='None', cmap=cm.gray )
show()
# uu = (250)-u
# plt.plot(uu)
# plt.show()
##############################################
"Calculate and print Z=X-u"
print('Z')
Z=X-u
print(Z)
print('shape: ', Z.shape)
print("Check Mins and Max Values",np.amin(Z),np.amax(Z))
print('\n\n\n')
##############################################
"Test the mean of Z to be zero (or very close to zero)"
# uu=np.mean(Z,axis=0); print(uu)
print('\n\n\n')
##############################################
"Calculate and print the Covariance of Z"
print('C')
C=np.cov(Z,rowvar=False)
print(C)
print('shape: ', C.shape)
print('\n\n\n')
##############################################
"Check that C is it's own inverse"
# if C.all()==C.T.all():
#     print("The Covariance matrix IS equal to it's inverse")
# else:
#     print(False)
# print('\n\n\n')
##############################################
"Plot the 728x728 covariance matrix"
plt.imshow(C,interpolation='None', cmap=cm.gray )
show()
##############################################
'Calculate Eigen Vectors'
print('l')
[L,V]=LA.eigh(C)
print(L)
print('shape: ', L.shape)
print('\n\n\n')
print('V')
print(V)
print('shape: ', V.shape)
print('\n\n\n')
##############################################
"""Test  if rows, or collumns are Eigen Vectors"""
# print('\n','Test Collumn: ')
L=np.flipud(L);V=np.flipud(V.T)
row=V[0,:]
# print(np.dot(C,row)/(L[0]*row))
# print('\n\n\n')
##############################################
"""Check Orthogonality of two rows of V"""
# should_be_zero = 0
# for i, j in enumerate(V[0,:]):
#     should_be_zero = should_be_zero + V[0,:][i]*V[1,:][j]
# print("The sum of two eigen vectors: \n (should be zero)\n")
# print(should_be_zero)
# print('\n\n\n')
##############################################
"""Check that two rows of V are Eigen vectors DOES NOT MAKE SENSE"""
# rowcol = 0
# for i, j in enumerate(V[0,:]):
#     rowcol = rowcol + V[0,:][i]*C[:,0][j]
# normalized = np.sqrt(rowcol*rowcol)
# print(rowcol, "\n")
# print(normalized)
# ei = list()
# for l, k in enumerate(V[0,:]):
#     ei = ei + ((rowcol[l]/normalized)-V[0,:][l])**2
# print(ei)
##############################################
"""Plot eigen vector"""
# plt.imshow(V[0,:].reshape(28, 28),interpolation='None', cmap=cm.gray )
# show()
##############################################
"""Calculate and print P check that the mean of P is (close to) zero"""
P = np.dot(Z,V[:,:2])
# print(np.mean(P)) """Check if the mean of P is zero"""
# print('\n')
print('P')
print(P)
print('shape: ', P.shape)
print('\n\n\n')
##############################################
# """Calculate and print R"""
# R = np.dot(P,V)
# print('R-Z')
# print(R-Z)
# print('shape: ', R.shape)
# print('\n\n\n')
##############################################
# """Recover X"""
# Xrec=R+u
# print('Xrec')
# print(Xrec-X) #X is recovered since Xrec-X is seen to contain very small values
# print('shape: ', Xrec.shape)
# print('\n\n\n')
##############################################
# """Dimesion Reduction 1"""
# Xrec1=(np.dot(P[:,0:1],V[0:1,:]))+u
# # print(Xrec1) #Reconstruction using 2 components
# print('Reduced 1')
# print('\n','min: ',np.amin(Xrec1), '\n max: ', np.amax(Xrec1))
# print('shape: ', Xrec1.shape)
# print('\n\n\n')
##############################################
# """Dimesion Reduction 2"""
# Xrec2=(np.dot(P[:,0:2],V[0:2,:]))+u
# # print(Xrec2) #Reconstruction using 2 components
# print('Reduced 2')
# print('\n','min: ',np.amin(Xrec2), '\n max: ', np.amax(Xrec2))
# print('shape: ', Xrec2.shape)
# print('\n\n\n')
##############################################
# """Plot reconstr"""
# plt.scatter(P)
# plt.show()
# fig, ax = plt.subplots()
# x1 = Xrec1
# y2 = Xrec2
# ax.scatter(x1, color='r' , alpha=.4)
# ax.scatter (y2, color='b', alpha=.4)
# plot_url = py.plot_mpl(fig, filename="mpl-complex-scatter")
##############################################
# plt.imshow(Xrec2[0,:].reshape(28, 28),interpolation='None', cmap=cm.gray )
# show()
# plt.imshow(Xrec2[1,:].reshape(28, 28),interpolation='None', cmap=cm.gray )

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.scatter(P[:,0],P[:,1],color='blue',s=5,edgecolor='none')
plt.show()
# ax1.set_aspect(1./ax1.get_data_ratio()) # make axes square






