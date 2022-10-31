import numpy as np

class EMatrix():
    def __init__(self, F, K) -> None:
        self.F = F
        self.K = K

    def getEssentialMatrix(self):

        E = np.dot(np.dot(self.K.T, self.F), self.K)
        u,D,v_t = np.linalg.svd(E)

        # Enforcing Rank 2 constraint
        D = np.diag([1,1,0])

        # Newly caluclated E amtrix
        E = np.dot(np.dot(u,D),v_t)
        return E
