import numpy as np

def Camera_poses(E):

    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    U,D,V_t = np.linalg.svd(E)

    T_1=U[:, 2]
    T_2=-U[:, 2]

    R_1=(np.dot(U, np.dot(W, V_t)))
    R_2=(np.dot(U, np.dot(W.T, V_t)))
    
    C1, R1 = pose_correction(T_1, R_1)
    C2, R2 = pose_correction(T_1, R_2)
    C3, R3 = pose_correction(T_2, R_1)
    C4, R4 = pose_correction(T_2, R_2)

    C = np.array([C1, C2, C3, C4])
    R = np.array([R1, R2, R3, R4])

    return C, R

def pose_correction(T, R):
    if np.linalg.det(R) > 0:
        return T,R
    else:
        return -T, -R