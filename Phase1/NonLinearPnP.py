from scipy.spatial.transform import Rotation as trans
import scipy.optimize as optimizer
import numpy as np
def reprojection_error(params,K,X,x):
    
    #initializing paramters
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    C = params[0:3]
    C = C.reshape(-1, 1)
    R = params[3:7]
    
    #calculating projection matrix
    transform= trans.from_quat([R[0], R[1], R[2], R[3]])
    R = transform.as_dcm()
    P= np.dot(np.dot(K, R), np.hstack((np.identity(3), -C)))

    #reprojection error calculation
    x_hat = np.dot(P, X)
    error=((x_hat[0] - x[0])**2) + ((x_hat[1] - x[1])**2)
    return error

def PNP_nonlinear(x_list,X_list,K,R_updated,C):

    '''
    Inputs:
        x_list : A list of all image coordinates of size nx2
        X_list : A list of all 3D coordinates  of size nx3
        K      : Intrinsic camera matrix of size 3x3
        R      : Rotation matrix of size 3x3
        C      : Camera pose of size 3x1

    Outputs:
        R      : Rotation matrix of size 3x3
        C      : Camera pose of size 3x1
    '''

	# extract image points
    # poses_non_linear = {}

	# extract point correspondences of given camera
	# corresp = corresp_2d_3d
	# x_list = corresp[:, 0:2]
	# X_list = corresp[:, 2:]

	# make the projection projection matrix
	# R = pose[:, 0:3]
	# C = pose[:, 3]
	# C = C.reshape((3, 1))

	# convert rotation matrix to quaternion form
    transform=trans.from_matrix(R_updated)
    q0=transform.as_quat()
    params= [C[0], C[1], C[2], q0[0], q0[1], q0[2], q0[3]]

    #optimization
    param_new = optimizer.least_squares(fun=reprojection_error, method="dogbox", x0=params, args=[K, X_list, x_list])
    C_updated = param_new.x[0:3]
    assert len(C_updated) == 3, "Translation Nonlinearpnp error"
    
    #reconvert back to rotation matrix
    R_updated = param_new.x[3:7]
    transform_r = trans.from_quat([R_updated[0], R_updated[1], R_updated[2], R_updated[3]])
    R_updated = transform.as_dcm()
    
    return R_updated,C_updated