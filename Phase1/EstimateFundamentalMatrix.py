import numpy as np

class FMatrix():
    def __init__(self, points) -> None:
        self.points = points

    def estimateFMatrix(self):
        A = np.array()
        for pts in self.points:
            x1, y1, x2, y2 = pts[0], pts[1], pts[2], pts[3]

            np.append(A, [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])

        u, s, vt = np.linalg.svd(A)
        # Ax = 0

        v = vt.T
        x = v[:, -1]
        # last column of the vt is the solution
        # same as autocalib

        F = np.reshape(x, (3,3)).T

        U, S, VT = np.linalg.svd(F)
        S[2] = 0

        F = U.dot(np.diag(S)).dot(VT)

        return F



