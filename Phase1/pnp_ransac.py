import LinearPnP
import numpy as np
import random


def reprojection_error_estimation(x,X,P,ret=False):
    '''
    Inputs:
    x - 2D points , size: 3X1
    X - 3D points, size: 3X1
    P - Projection matrix, size: 3X4

    Outputs:
    error - value
    x_hat - (conditionla output) 
    '''

    X = np.hstack((X, np.ones((X.shape[0], 1))))
    X = X.T

    # Normalizing projected points
    x_hat = np.dot(P, X)
    x_hat = x_hat.T
    normalization_factor=x_hat[:,2]
    x_hat=x_hat/normalization_factor
    x_hat=x_hat[:,:2]
    
    # reprojjection error
    error= x - x_hat
    error= error**2
    error= np.sqrt(np.sum(error, axis=1))

    if(ret):
        return error, x_hat

    return error


def pnp_ransac(correspondences,K, thresh = 20,max_inliers=0):
    '''
    Inputs:
    x_list - 2D points , size: 3X1
    X_list - 3D points, size: 3X1
    K - Intrinsic matrix, size: 3X3

    Outputs:
    C - Translation vector of size 3x1
    R - Rotation matrix size 3x3
    '''

    random.seed(26)

    # RANSAC algorithim
    for i in range(10000):

        # choose 6 random points and get linear pnp estimate
        sample = np.array(random.sample(correspondences, 6), np.float32)
        sample_x=sample[:,:2]
        sample_X=sample[:,2:]

        R, C = LinearPnP.PNP_linear(sample_x,sample_X, K)

        # form the projection matrix
        C = np.reshape(C,(3, 1))
        I = np.identity(3)
        P = np.hstack((I, -C))
        P = np.dot(K, np.dot(R, P))

        error = reprojection_error_estimation(sample_x,sample_X,P)
        locs = np.where(error < thresh)[0]
        n = np.shape(locs)[0]
        if n > max_inliers:
            max_inliers = n
            inliers = correspondences[locs]
            R_best = R
            C_best = C

    P_best = np.hstack((R_best, C_best))
    return P_best, inliers