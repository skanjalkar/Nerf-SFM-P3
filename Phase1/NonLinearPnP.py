from scipy.spatial.transform import Rotation as trans
import scipy.optimize as optimizer
import numpy as np
def optimizer_function(params,K,x,X):
    
    #initializing paramters
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    C = params[0:3]
    C = C.reshape(-1, 1)
    R = params[3:7]
    
    #calculating projection matrix
    transform= trans.from_quat([R[0], R[1], R[2], R[3]])
    R = transform.as_matrix()
    P= np.dot(np.dot(K, R), np.hstack((np.identity(3), -C)))
    reprojection_error_list=[]
    for a_x,a_X in zip(x,X):
    #reprojection error calculation
        error=err(P,a_x,a_X)
        reprojection_error_list.append(error)
    reprojection_error_list=np.array(reprojection_error_list)
    reprojection_error_list=reprojection_error_list.reshape(reprojection_error_list.shape[0],)
    return reprojection_error_list

def err(P, x, X):
    
    x_hat = np.dot(P, X)
    norm=x_hat[2]
    x_hat=x_hat/norm
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
	
    # convert rotation matrix to quaternion form
    transform=trans.from_matrix(R_updated)
    q0=transform.as_quat()
    params= [C[0], C[1], C[2], q0[0], q0[1], q0[2], q0[3]]


    #optimization
    param_new = optimizer.least_squares(fun=optimizer_function, method="dogbox", x0=params, args=[K, x_list, X_list])
    C_updated = param_new.x[0:3]
    assert len(C_updated) == 3, "Translation Nonlinearpnp error"
    
    #reconvert back to rotation matrix
    R_updated = param_new.x[3:7]
    transform_r = trans.from_quat([R_updated[0], R_updated[1], R_updated[2], R_updated[3]])
    R_updated = transform_r.as_matrix()
    
    return R_updated,C_updated.reshape(3,1)