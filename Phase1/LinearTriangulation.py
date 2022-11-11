import numpy as np


def skew(self, x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, x[0]], [x[1], x[0], 0]])


# def LinearTriangulation(self, C1, R1, C2, R2, x1, x2):
#     n_points = x1.shape[0]

#     C1 = np.reshape(C1, (3, 1))
#     P1 = np.dot(self.K, np.dot(R1, np.hstack((np.identity(3), -C1))))

#     C2 = np.reshape(C2, (3, 1))
#     P2 = np.dot(self.K, np.dot(R2, np.hstack((np.identity(3), -C2))))
#     # pry()
#     # print(P2.shape)
#     X1 = np.hstack((x1, np.ones((n_points, 1))))
#     X2 = np.hstack((x2, np.ones((n_points, 1))))

#     X = np.zeros((n_points, 3))

#     for i in range(n_points):
#         skew1 = self.skew(X1[i, :])
#         skew2 = self.skew(X2[i, :])
#         A = np.vstack((np.dot(skew1, P1), np.dot(skew2, P2)))
#         u, D, v = np.linalg.svd(A)
#         normalized_3D_coordinates = np.array(v[-1]/v[-1, -1]).flatten()
#         # x = np.reshape(x, (len(x), -1)
#         X[i, :] = normalized_3D_coordinates[0:3].T
#     return X

def LinearTrinagulation(self, P2, C2, R2, K, x1,x2):
    '''
    Inputs:
        P2 : Aprojection matrix of first camera
        C2 : A 3x1 matrix of cameras translation
        R2 : Rotation matrix of camera of size 3x3
        K  : Camera instrinsinc matrix of size 3x3
        x_cur_x_new : A 2D -2D correspondence between a pair of images of size nx4 : A row looks like [x1,y1,x2,y2]

    Outputs:
        R      : Rotation matrix of size 3x3
        C      : Camera pose of size 3x1

    '''


    ones = np.ones((x1.shape[0], 1))
    x1 = np.hstack((x1, ones))
    x2 = np.hstack((x2, ones))

    I = np.identity(3)
    M2 = np.hstack((I, -C2))
    M2 = np.dot(K, np.dot(R2, M2))

    xList = []

    for p1, p2 in zip(x1, x2):
        A = [p1[0]*P2[2, :] - P2[0, :]]
        A.append(p1[1]*P2[2, :] - P2[1, :])
        A.append(p2[0]*M2[2, :] - M2[0, :])
        A.append(p2[1]*M2[2, :] - M2[1, :])

        A = np.array(A)

        U, S, VT = np.linalg.svd(A)
        v = VT.T
        X = v[:, -1]

        X = X/X[3]
        X = X[:3]

        X = np.array(X)
        X = X.reshape((3, 1))

        xList.append(X)
    xList = np.array(xList)
    return xList