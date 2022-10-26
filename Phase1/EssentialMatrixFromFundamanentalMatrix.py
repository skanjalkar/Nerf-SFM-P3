import numpy as np
def essential_matrix(F, K):

    E = np.dot(np.dot(K.T, F), K)
    u,D,v_t = np.linalg.svd(E)
    
    # Enforcing Rank 2 constraint
    D = np.diag([1,1,0])
    
    # Newly caluclated E amtrix
    E = np.dot(np.dot(u,D),v_t)
    return E
