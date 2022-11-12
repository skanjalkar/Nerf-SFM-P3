import argparse
import numpy as np
from Helper.ImageHelper import *
from Helper.Misc import x_X_map_creator
from Helper.createPairMatches import *
from GetInliersRANSAC import *
from EssentialMatrixFromFundamanentalMatrix import *
from ExtractCameraPose import *
from DisambiguateCameraPose import *
from NonlinearTriangulation import Non_Linear_Triangulation
from pnp_ransac import pnp_ransac
from pnp_ransac import reprojection_error_estimation
from NonLinearPnP import PNP_nonlinear
from BundleAdjustment import bundle_adjustment
from LinearTriangulation import LinearTrinagulation
from pathlib import Path, PureWindowsPath
from Helper.Misc import plot_3D
from Helper.Misc import plot_after_BA


cwd = Path.cwd()

def readCalibrationMatrix(path, windows=False):
    '''Read the calibration matrix'''

    path=os.path.join(path,"calibration.txt")
    with open(path, 'r') as f:
        contents = f.read()

    K = []
    for i in contents.split():
        K.append(float(i))

    K = np.array(K)
    K = np.reshape(K, (3,3))
    return K

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--data_path', default=os.path.join(os.getcwd(),"Data","MatchingData"), help='Data path')
    Parser.add_argument('--findImgPair', default=False, type=bool, help='To get the matches for all the pairs')
    Parser.add_argument('--os', default=True, type=bool, help="If os is Linux or mac type False")
    Args = Parser.parse_args()
    data_path = Args.data_path
    findImgPair = Args.findImgPair
    windowsos = Args.os
    # print(sys.path)
    if findImgPair:
        createMatchestxt(os.path.join(os.getcwd(),"Data"))
    # get the camera calibration matrix
    K = readCalibrationMatrix(data_path, windowsos)

    P1 = np.dot(K, np.hstack((np.identity(3), np.zeros((3,1)))))
    R1, C1 = P1[:3, :3], P1[:, -1]
    # pry()
    imgHelper = ImageHelper(data_path)
    images = imgHelper.readImages()
    # initial points(before ransac)
    matchPoints = imgHelper.readPoints("matches12.txt")
    matchPoints = np.array(matchPoints, np.float32)

    # RANSAC
    img1, img2 = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY), cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY)
    ransacObj = RANSAC()
    do_RANSAC=False
    if do_RANSAC:
        inlierPoints, outlierPoints, bestF, img1Pts, img2Pts = ransacObj.getInliersRansac(matchPoints)
        np.save("inlierPoints",inlierPoints)
        np.save("outlierPoints",outlierPoints)
        np.save("bestF",bestF)

    else:
        inlierPoints=np.load(os.path.join(os.getcwd(),"inlierPoints.npy"))
        outlierPoints=np.load("outlierPoints.npy")
        bestF=np.load("bestF.npy")
    # imgHelper.plotInliers(images[0], images[1], inlierPoints, "Inliers", False)
    # imgHelper.plotOutliers(images[0], images[1], outlierPoints)


    # Essential matrix
    eObj = EMatrix(bestF, K)
    E = eObj.getEssentialMatrix()

    # Camera Pose
    cameraPoseObj = CameraPose(E)
    potentailC2, potentailR2 = cameraPoseObj.cameraPoses()



    # LinearTriangulation
    disObj = Disambiguate(inlierPoints, K)
    bestX, bestC, bestR, index = disObj.disambiguateCameraPose(potentailC2, potentailR2, P1,K)
    bestX.reshape(-1,3)
    #Storing error values
    mean_proj_error = {}
    P2=np.dot(np.dot(K, bestR), np.hstack((np.identity(3), -bestC)))
    reproj_errors = reprojection_error_estimation(inlierPoints[:, 2:4],bestX,P2)
    print("2lin",np.mean(reproj_errors))
    mean_proj_error[2] = [('Linear_Triangulation', np.mean(reproj_errors))]

    #plotting linear triangulation

    #------------------------------------------------
    #NON Linear trinagulation
    print("Finished Linear triangulation\nBeginning Non Linear triangulation")
   

    #saving values
    save=False
    if save:
        X_list =  Non_Linear_Triangulation(P1, P2, bestX, inlierPoints)
        np.save("X_list_nonlin",X_list)
    else:
        X_list=np.load(os.path.join(os.getcwd(),"X_list_nonlin.npy"))

    # storing error values
    reproj_errors = reprojection_error_estimation(inlierPoints[:, 2:4],X_list,P2 )
    mean_proj_error[2] = [('NonLinear_Triangulation', np.mean(reproj_errors))]


    #plotting non-linear triangulation
    plotHelper = Plot()
    plotHelper.plotTriangle(X_list, bestC, bestR, index)
    # plt.xlim(-5, 5)
    # plt.ylim(-5, 5)
    # plt.show()



    #----------------------------------------------
    #Here we start prepping for PNP ransac
    print("Finished NonLinear triangulation\nPrepping for PNP")


    # Storing all camera poses in a dictionary
    camera_poses_dict = {}
    camera_poses_dict[1] = np.identity(4)[0:3, :]
    camera_poses_dict[2] = np.hstack((bestR, bestC.reshape(3,1)))


    image_correspondences = [[2, 3], [3, 4], [4, 5]]
    x_X_map_dict = {} #This is in the format x_X_map[1]= [x1,y1,X,Y,Z]

    # In this section we are trying to map each image's 2D points(x) with a global set of 3D points (X)

    #For image 1
    x_X_image_1 = inlierPoints[:, 0:2]
    x_X_image_1 = np.hstack((x_X_image_1, X_list))
    x_X_map_dict[1] = x_X_image_1

    # For image 2
    x_X_image_2 = inlierPoints[:, 2:4]
    x_X_image_2 = np.hstack((x_X_image_2, X_list))
    x_X_map_dict[2] = x_X_image_2

    # add the 3d points to X_set
    X_set=np.copy(X_list)


    x_Xindex_mapping = {} # a dictionary where key: iamge no val= nx3 matrix (each row: [x,y,X_index])
    x_Xindex_mapping[1] = zip(x_X_map_dict[1][:, 0:2], range(X_set.shape[0]))
    x_Xindex_mapping[2] = zip(x_X_map_dict[2][:, 0:2], range(X_set.shape[0]))

    # estimate pose for the remaining cams
    for _, a_correspondence in enumerate(image_correspondences):
        print("......................................")
        print(f"Estimating pose for Picture {a_correspondence[1]}")
        print("......................................")
        current_image = a_correspondence[0]
        new_img_num = a_correspondence[1]
        print(f"Estimating pose for Picture {new_img_num}")

        #Calculating projection Matrix of current image
        R_non_pnp = camera_poses_dict[current_image][:, 0:3]
        C_non_pnp = camera_poses_dict[current_image][:, 3].reshape((3, 1))
        P_current=np.dot(np.dot(K, R_non_pnp), np.hstack((np.identity(3), -C_non_pnp)))

        # get the 2d-3d correspondences for the 1st ref image
        x_X_cur_image = x_X_map_dict[current_image]

        # Get 2d -2d correpsondecnes for cur_image and next_image. of the format [x_cur,y_cur,x_next,y_next]
        correspondence_file_name=f"matches{current_image}{new_img_num}.txt"
        xcur_xnext=imgHelper.readPoints(correspondence_file_name)
 
        xcur_xnext=xcur_xnext[:,:4]
        print(xcur_xnext.shape)
        

        #  NOW WE GET : x_X mapping for new image
        #x_X_new_image is of the format [x1,y1,X,Y,Z]
        x_X_new_image, remaining_2d_2d = x_X_map_creator(x_X_cur_image, xcur_xnext)
        


        ##------------PnP RANSAC----------------------------
        print("------\nStarting PnP RANSAC ")
        # x_list_new_image=x_X_new_image[:2,:]
        # X_list_new_image=x_X_new_image[2:,:]
        pose_pnp_ransac, pnp_inlier_corresp = pnp_ransac(x_X_new_image,K, thresh=200)
        R_pnp = pose_pnp_ransac[:, 0:3]
        C_pnp = pose_pnp_ransac[:, 3].reshape(-1,1)
        P_pnp = np.dot(np.dot(K, R_pnp), np.hstack((np.identity(3), -C_pnp)))
        x_pnp = x_X_new_image[:, 0:2]
        X_pnp = x_X_new_image[:, 2:5]

        # plot_3D(X_pnp,C_pnp,"after pnp")

        #error estimation and storing
        reproj_errors = reprojection_error_estimation(x_pnp, X_pnp,P_pnp)
        mean_proj_error[new_img_num] = [('Linear_PnP', np.mean(reproj_errors))]


        ###---------------NON LINEAR PNP-------------------
        print("------\nperforming Non-linear PnP to obtain optimal pose")


        R_non_pnp,C_non_pnp = PNP_nonlinear(x_pnp,X_pnp,K, R_pnp, C_pnp)
        P_non_pnp=np.dot(np.dot(K, R_non_pnp), np.hstack((np.identity(3), -C_non_pnp)))

        # error is calculated for the same set of image and 3d points but with refined pose
        reproj_errors = reprojection_error_estimation(x_pnp,X_pnp,P_non_pnp)
        mean_proj_error[new_img_num].append(('NonLinPnP', np.mean(reproj_errors)))

        # plot_3D(X_pnp,C_non_pnp,"after non pnp")
        ###---------------LINEAR TRIANGULATION FOR REMAINIG POINTS-------------------
        print("------\nPerforming Linear Triangulation on remaining points")
        # find the 2d-3d mapping for the remaining image points in the new image by doing triangulation

        x_cur_remaining=remaining_2d_2d[:,0:2]
        x_new_remaining=remaining_2d_2d[:,2:4]
        X_lin_tri_remaining = LinearTrinagulation(P_current, C_non_pnp, R_non_pnp, K, x_cur_remaining,x_new_remaining)
        X_lin_tri_remaining = X_lin_tri_remaining.reshape((remaining_2d_2d.shape[0], 3))


        print(f"{x_X_new_image.shape} points before adding remaining correspondences")
        x_X_new_image_linear = np.vstack((x_X_new_image, np.hstack((x_new_remaining, X_lin_tri_remaining))))
        print(f"{x_X_new_image_linear.shape} points after adding remaining correspondences")

        # plot_3D(X_lin_tri_remaining,C_non_pnp,"after lin tri remainig points")

        # error calculation and storing
        reproj_errors = reprojection_error_estimation(x_X_new_image_linear[:, 0:2],x_X_new_image_linear[:, 2:], P_non_pnp)
        mean_proj_error[new_img_num].append(('LinTri_compleyte points', np.mean(reproj_errors)))

    ##---------------NON LINEAR TRIANGULATION FOR REMAINIG POINTS-------------------
        print("------\nNon Liner triangulation for remaining correspondences")
        X_non_lin_tri = Non_Linear_Triangulation(P_current, P_non_pnp, X_lin_tri_remaining, remaining_2d_2d)
        X_non_lin_tri = X_non_lin_tri.reshape((remaining_2d_2d.shape[0], 3))

        x_X_new_image_non_linear = np.vstack((x_X_new_image, np.hstack((remaining_2d_2d[:, 2:4], X_non_lin_tri))))

        # error calculation and storing
        x_non_lintriang_all = x_X_new_image_non_linear[:, 0:2]
        X_non_lintriang_all = x_X_new_image_non_linear[:, 2:]
        reproj_errors = reprojection_error_estimation(x_X_new_image_non_linear[:,:2],x_X_new_image_non_linear[:,2:], P_non_pnp)
        mean_proj_error[new_img_num].append(('NonLinTri_complete_points', np.mean(reproj_errors)))

        # plot_3D(X_non_lintriang_all,C_non_pnp,"after non linear triangulation of all points")
        # store the current pose after non linear pnp
        camera_poses_dict[new_img_num] = np.hstack((R_non_pnp, C_non_pnp.reshape((3, 1))))

        # plotting reprojected points
        # plot_funcs.plot_reproj_points(images[new_img_num-1], new_img_num, np.float32(pts_img_all), np.float32(pts_img_reproj_all), save=True)
        # print("plotting all the camera poses and their respective correspondences\n")
        x_X_map_dict[new_img_num] = np.hstack((x_non_lintriang_all, X_non_lintriang_all))
        # plot_funcs.plot_camera_poses(camera_poses_dict, x_X_map_dict, save=True)


        ##-----------------------BUNDLE ADJUSTMENT-----------------------
        # initilaization for BA
        index_start = X_set.shape[0]
        index_end = X_set.shape[0] + X_non_lintriang_all.shape[0]
        x_Xindex_mapping[new_img_num] = zip(x_non_lintriang_all, range(index_start, index_end))
        X_set = np.append(X_set, X_non_lintriang_all, axis=0)

        
        #BA implemntation refereced from https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html 
        print("------\nStarting Bundle adjustment ")
        pose_set_opt, X_set_opt = bundle_adjustment(camera_poses_dict, X_set, x_Xindex_mapping, K)
        print("keys --> {}".format(pose_set_opt.keys()))
        # compute reproj error after BA
        R_ba = pose_set_opt[new_img_num][:, 0:3]
        C_ba = pose_set_opt[new_img_num][:, 3]
        X_all_ba = X_set_opt[index_start:index_end].reshape((x_non_lintriang_all.shape[0], 3))
        P_ba = np.dot(np.dot(K, R_ba), np.hstack((np.identity(3), -C_ba.reshape(3,1))))
        reproj_errors = reprojection_error_estimation(x_non_lintriang_all, X_all_ba,P_ba)
        mean_proj_error[new_img_num].append(('BA', np.mean(reproj_errors)))


        # make X_set = X_set_opt and send it for further iterations...
        X_set = X_set_opt

        # Save the optimal correspondences (2D-3D) and the optimal poses for next iteration
        x_X_map_dict[new_img_num] = np.hstack((x_non_lintriang_all, X_all_ba))
        # pose_set[new_img_num] = np.hstack((R_ba, C_ba.reshape((3, 1))))
        camera_poses_dict = pose_set_opt
        print(mean_proj_error[new_img_num])
        print("......................................")
        plot_after_BA(camera_poses_dict,X_set)
    
    print(mean_proj_error)
if __name__ == "__main__":
    main()