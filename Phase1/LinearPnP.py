import numpy as np
def skew(x):
    x=x.flatten()
    y = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    return y
def PNP_linear(x_list,X_list,K):
    '''
    Inputs:
    x - 2D points , size: 3X1
    X - 3D points, size: 3X1
    K - Intrinsic matrix, size: 3X3

    Outputs:
    C - Translation vector of size 3x1
    R - Rotation matrix size 3x3
    '''

    #Initializing A matrix for SVD

    #############################need to inialize this properly
    A = np.empty((0, 12), np.float32)
    
    for x,X in zip(x_list,X_list):
        # normalizing coordinates
        x=np.array([x[0],x[1],1]).reshape(3,-1)
        x_normalized = np.dot(np.linalg.inv(K),x)
        x_normalized = x_normalized/x_normalized[2]

        X = np.append(X, 1)
        X=X.reshape(-1,4)
        #Creating A matrix (refer to report)
        row_1=np.hstack((X,np.zeros((1,8))))
        row_2=np.hstack((np.zeros((1,4)),X,np.zeros((1,4))))
        row_3=np.hstack((np.zeros((1,8)),X))

        # row_1=np.concatenate((X.reshape(-1,1),np.zeros((1,8))))
        # row_2=np.concatenate((np.zeros((1,4)),X.T,np.zeros((1,4))))
        # row_3=np.concatenate((np.zeros((1,8)),X.T))

        A_one_point=np.vstack((row_1,row_2,row_3))
        A_one_point=np.dot(skew(x),A_one_point)
        A=np.vstack((A,A_one_point))
       
    A = np.float32(A)
    U,D,V_t = np.linalg.svd(A)

    V = V_t.T

    # Finding the projection matrix
    P = V[:, -1]
    P = P.reshape((3, 4))

    # Getting teh rotation and translation matrices
    R = P[:, 0:3]
    T = P[:, 3]
    R = R.reshape((3, 3))
    T = T.reshape((3, 1))

    # Recacalculating R to enforce the orthonormal rule
    U,D,V_t = np.linalg.svd(R)
    R= np.dot(U, V_t)

    # Checking for determinant value of r
    if np.linalg.det(R) < 0:
        R = -R
        T = -T

    C = -np.dot(R.T, T)
    return R, C
# X=np.ones((4,1))
# x=np.ones((3,1))*2
# zeros=np.zeros((1,4))

# a=np.hstack((X.T,zeros))

# row_1=np.hstack((X.T,np.zeros((1,8))))
# row_2=np.hstack((np.zeros((1,4)),X.T,np.zeros((1,4))))
# row_3=np.hstack((np.zeros((1,8)),X.T))
# A=np.vstack((row_1,row_2,row_3))
# A=np.dot(skew(x),A)
