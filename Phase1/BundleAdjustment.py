import numpy as np
from scipy.optimize import least_squares
from BuildVisibilityMatrix import sparsity_matrix
import time


from scipy.spatial.transform import Rotation as trans

def bundle_adjustment(pose_set, X_points, x_Xindex_mapping, K):
    '''
    Input arguments--
    
    pose_set         : It is a dictionary containing where the key= img no, val=3x4 matrix (where first 3 columns are R and the last column is C)
    X_points         : A nx3 matrix where each row is [X,Y,Z]
    x_Xindex_mapping : a dictionary where key: iamge no val= nx3 matrix (each row: [x,y,X_index])
    K                : Intrinsinc camera matrix

    Outputs ---

    optimized pose set
    optimized X set
    '''

    #getting parameters to optimize on
    n_cam, n_3d, indices_3d_pts, img_pts_2d, indices_cam, x0 = optimization_parameters(pose_set, X_points, x_Xindex_mapping)

    #getting visibility matrix
    A=sparsity_matrix(n_cam, n_3d, indices_cam, indices_3d_pts)

    begin = time.time()
    result = least_squares(fun=optimizer, x0=x0, jac_sparsity=A, verbose=2, x_scale='jac',ftol=1e-4, method='trf', args=(n_cam, n_3d, indices_3d_pts, img_pts_2d, indices_cam, K))
    end = time.time()
    print("Time taken for optimization: {} seconds".format(begin-end))

    camera_params = result.x[:n_cam*7].reshape((n_cam, 7))
    X_world_all_opt = result.x[n_cam*7:].reshape((n_3d, 3))
    optimized_pose_set = {}
    
    i = 1
    for a_camera_cam in camera_params:
        transform= trans.from_quat(a_camera_cam[:4])
        R = transform.as_matrix()
        C = a_camera_cam[4:].reshape((3, 1))
        optimized_pose_set[i] = np.hstack((R, C))
        i += 1

    return optimized_pose_set, X_world_all_opt

def optimization_parameters(pose_set, X_points, map_2d_3d):

    #initilizing paramters
    params = np.empty(0, dtype=np.float32)
    pts_3D_indices = np.empty(0, dtype=int)
    pts_2D = np.empty((0, 2), dtype=np.float32)
    camera_indices = np.empty(0, dtype=int)
    n_cameras = max(pose_set.keys())

    # Appending parametrs for each pose
    for i in pose_set.keys():

        transform=trans.from_matrix(pose_set[i][:, 0:3])
        quaternion=transform.as_quat()
        camera_point = pose_set[i][:, 3]        
        params = np.append(params, quaternion.reshape(-1), axis=0)
        params = np.append(params, camera_point, axis=0)

        #appending 2D and 3D points
        for j in map_2d_3d[i]:
            pts_3D_indices = np.append(pts_3D_indices, [j[1]], axis=0)
            pts_2D = np.append(pts_2D, [j[0]], axis=0)
            camera_indices = np.append(camera_indices, [i-1], axis=0)

    params = np.append(params, X_points.flatten(), axis=0)
    n_3d_points = X_points.shape[0]

    return n_cameras, n_3d_points, pts_3D_indices, pts_2D, camera_indices, params

def optimizer(params, n_cam, n_3d, indices_3d_pts, img_pts_2d, indices_cam, K):

    param_cam = params[:n_cam*7].reshape((n_cam, 7))
    param_cam=param_cam[indices_cam]
    
    world_pts_3d = params[n_cam*7:].reshape((n_3d, 3))
    world_pts_3d = world_pts_3d[indices_3d_pts]
    ones = np.ones((world_pts_3d.shape[0], 1))
    world_pts_3d = np.hstack((world_pts_3d, ones))

    pt_img_proj = np.empty((0, 2), dtype=np.float32)

    for i, X in enumerate(world_pts_3d):
        transform= trans.from_quat((param_cam[i, :4]))
        R = transform.as_matrix()
        C = param_cam[i, 4:]
        P= np.dot(np.dot(K, R), np.hstack((np.identity(3), -C)))
        X = X.reshape((4, 1))
        x = np.dot(P, X)
        x = x/x[2]
        x = x[:2]
        x = x.reshape((1, 2))
        pt_img_proj = np.append(pt_img_proj, x, axis=0)

    reproj_err = img_pts_2d - pt_img_proj
    return reproj_err.ravel()