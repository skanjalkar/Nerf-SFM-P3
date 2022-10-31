import numpy as np
import pry

class LinearTriangulation():
    def __init__(self, K) -> None:
        self.K = K

    def skew(self, x):
        return np.array([[0, -x[2], x[1]], [x[2], 0, x[0]], [x[1], x[0], 0]])

    def LinearTriangulation(self, C1, R1, C2, R2, x1, x2):


        n_points = x1.shape[0]

        C1 = np.reshape(C1, (3, 1))
        P1 = np.dot(self.K, np.dot(R1, np.hstack((np.identity(3), -C1))))

        C2 = np.reshape(C2, (3, 1))
        P2 = np.dot(self.K, np.dot(R2, np.hstack((np.identity(3), -C2))))
        # pry()
        # print(P2.shape)
        X1 = np.hstack((x1, np.ones((n_points, 1))))
        X2 = np.hstack((x2, np.ones((n_points, 1))))

        X = np.zeros((n_points, 3))

        for i in range(n_points):
            skew1 = self.skew(X1[i, :])
            skew2 = self.skew(X2[i, :])
            A = np.vstack((np.dot(skew1, P1), np.dot(skew2, P2)))
            u, D, v = np.linalg.svd(A)
            normalized_3D_coordinates = np.array(v[-1]/v[-1, -1]).flatten()
            # x = np.reshape(x, (len(x), -1)
            X[i, :] = normalized_3D_coordinates[0:3]

        return X