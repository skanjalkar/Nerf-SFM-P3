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

def readCalibrationMatrix(path):
    '''Read the calibration matrix'''
    print(path)
    with open(path+"calibration.txt", 'r') as f:
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
    Parser.add_argument('--data_path', default='../Data/MatchingData/', help='Data path')
    Parser.add_argument('--findImgPair', default=False, type=bool, help='To get the matches for all the pairs')
    Args = Parser.parse_args()
    data_path = Args.data_path
    findImgPair = Args.findImgPair

    if findImgPair:
        createMatchestxt(data_path)
    # get the camera calibration matrix
    K = readCalibrationMatrix(data_path)

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
    inlierPoints, outlierPoints, bestF, img1Pts, img2Pts = ransacObj.getInliersRansac(matchPoints)
    # pts1, pts2 = [], []
    # for i in range(len(inlierPoints)):
    #     img1x, img1y = int(inlierPoints[i][0]), int(inlierPoints[i][1])
    #     pts1.append((img1x, img1y))
    #     img2x, img2y = int(inlierPoints[i][2]), int(inlierPoints[i][3])
    #     pts2.append((img2x, img2y))
    # pts1 = np.array(pts1)
    # pts2 = np.array(pts2)
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    # lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,bestF)
    # lines1 = lines1.reshape(-1,3)
    # img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    # lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,bestF)
    # lines2 = lines2.reshape(-1,3)
    # img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

    # cv2.imshow('img3', img3)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imshow('img5', img5)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    imgHelper.plotInliers(images[0], images[1], inlierPoints, "Inliers", False)
    imgHelper.plotOutliers(images[0], images[1], outlierPoints)

    # Essential matrix
    eObj = EMatrix(bestF, K)
    E = eObj.getEssentialMatrix()

    # Camera Pose
    cameraPoseObj = CameraPose(E)
    potentailC2, potentailR2 = cameraPoseObj.cameraPoses()

    # LinearTriangulation
    disObj = Disambiguate(inlierPoints, K)
    bestX, bestC, bestR, index = disObj.disambiguateCameraPose(C1, R1, potentailC2, potentailR2, P1)

    #Storing error values
    mean_proj_error = {}
    P2=np.dot(np.dot(K, bestR), np.hstack((np.identity(3), -bestC)))
    reproj_errors = reprojection_error_estimation(inlierPoints[:, 2:4],bestX,P2,get_val=True )
    mean_proj_error[2] = [('Linear_Triangulation', np.mean(reproj_errors))]

    #plotting linear triangulation
    
    #------------------------------------------------
    #NON Linear trinagulation
    print("Finished Linear triangulation\nBeginning Non Linear triangulation")
    X_list =  Non_Linear_Triangulation(P1, P2, bestX, inlierPoints, K)

    # storing error values
    reproj_errors = reprojection_error_estimation(inlierPoints[:, 2:4],X_list,P2,get_val=True )
    mean_proj_error[2] = [('NonLinear_Triangulation', np.mean(reproj_errors))]


    #plotting non-linear triangulation



    #----------------------------------------------
    #Here we start prepping for PNP ransac
    print("Finished NonLinear triangulation\nPrepping for PNP")


    # Storing all camera poses in a dictionary
    camera_poses_dict = {}
    camera_poses_dict[1] = np.identity(4)[0:3, :]
    camera_poses_dict[2] = np.hstack((bestR, bestC.reshape(3,1)))

    
    image_correspondences = [[2, 3], [3, 4], [4, 5], [5, 6]]
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


    x_Xindex_mapping = {}
    x_Xindex_mapping[1] = zip(x_X_map_dict[1][:, 0:2], range(X_set.shape[0]))
    x_Xindex_mapping[2] = zip(x_X_map_dict[2][:, 0:2], range(X_set.shape[0]))

    # estimate pose for the remaining cams
    for _, a_correspondence in enumerate(image_correspondences):
        
        current_image = a_correspondence[0]
        new_img_num = a_correspondence[1]
        img_pair = str(current_image)+str(new_img_num)
        file_name = "ransac"+img_pair+".txt"

        #Calculating projection Matrix of current image
        R_non_pnp = camera_poses_dict[current_image][:, 0:3]
        C_non_pnp = camera_poses_dict[current_image][:, 3].reshape((3, 1))
        P_current=np.dot(np.dot(K, R_non_pnp), np.hstack((np.identity(3), -C_non_pnp)))

        # get the 2d-3d correspondences for the 1st ref image
        x_X_cur_image = x_X_map_dict[current_image]

        # Get 2d -2d correpsondecnes for cur_image and next_image. of the format [x_cur,y_cur,x_next,y_next]
        correspondence_file_name=f"matches{current_image}{new_img_num}"
        xcur_xnext=imgHelper.readPoints(correspondence_file_name)
        xcur_xnext=xcur_xnext[:4,:]
        
        
        # now we try to get x_X mapping for new image
        #x_X_new_image is of the format [x1,y1,X,Y,Z]
        x_X_new_image, remaining_2d_2d = x_X_map_creator(x_X_cur_image, xcur_xnext)
        print("shape of 2d-3d correspondences {}".format(np.shape(x_X_new_image)))

        ##.............................PnP RANSAC...........................'''
        print("PnP RANSAC to refine the poses")
        # x_list_new_image=x_X_new_image[:2,:]
        # X_list_new_image=x_X_new_image[2:,:]
        pose_pnp_ransac, pnp_inlier_corresp = pnp_ransac(x_X_new_image,K, thresh=200)
        R_pnp = pose_pnp_ransac[:, 0:3]
        C_pnp = pose_pnp_ransac[:, 3]
        P_pnp = np.dot(np.dot(K, R_pnp), np.hstack((np.identity(3), -C_pnp)))
        x_pnp = x_X_new_image[:, 0:2]
        X_pnp = x_X_new_image[:, 2:5]
        reproj_errors = reprojection_error_estimation(x_pnp, P_pnp, X_pnp)
        mean_proj_error[new_img_num] = [('LinPnP', np.mean(reproj_errors))]
        '''.............................Non-linear PnP...........................'''
        print("performing Non-linear PnP to obtain optimal pose")
        pose_non_linear = PNP_nonlinear(K, pose_pnp_ransac, pnp_inlier_corresp)
        R_non_pnp = pose_non_linear[:, 0:3]
        C_non_pnp = pose_non_linear[:, 3].reshape((3, 1))

        P_non_pnp=np.dot(np.dot(K, R_non_pnp), np.hstack((np.identity(3), -C_non_pnp)))
        
        # error is calculated for the same set of image and 3d points but with refined pose
        reproj_errors = reprojection_error_estimation(x_pnp,X_pnp,P_non_pnp)
        mean_proj_error[new_img_num].append(('NonLinPnP', np.mean(reproj_errors)))

        '''.............................Linear triangulation...........................'''
        print("performing Linear Triangulation to obtain 3d equiv for remaining 2d points")
        # find the 2d-3d mapping for the remaining image points in the new image by doing triangulation
        X_lin_tri = linear_triagulation(P_current, C_non_pnp, R_non_pnp, K, remaining_2d_2d)
        X_lin_tri = X_lin_tri.reshape((remaining_2d_2d.shape[0], 3))

        remaining_2d_3d_linear = remaining_2d_2d[:, 2:4]
        print("(linear)points before adding remaining corresp - {}".format(x_X_new_image.shape))
        new_img_2d_3d_linear = np.vstack((x_X_new_image, np.hstack((remaining_2d_3d_linear, X_lin_tri))))
        print("(linear)points after adding remaining corresp - {}".format(new_img_2d_3d_linear.shape))

        # print reprojection error after non-linear triangulation
        pts_img_all_linear = new_img_2d_3d_linear[:, 0:2]
        X_all_linear = new_img_2d_3d_linear[:, 2:]
        reproj_errors = compute_reproj_err_all(pts_img_all_linear, P_non_pnp, X_all_linear)
        mean_proj_error[new_img_num].append(('LinTri', np.mean(reproj_errors)))


        '''.............................Non-Linear triangulation...........................'''
        print("performing Non-Linear Triangulation to obtain 3d equiv for remaining 2d points")
        X_non_lin_tri = nonlinear_triang(P_current, P_non_pnp, X_lin_tri, remaining_2d_2d, K)
        X_non_lin_tri = X_non_lin_tri.reshape((remaining_2d_2d.shape[0], 3))

        remaining_2d_3d = remaining_2d_2d[:, 2:4]
        print("(Nlinear)points before adding remaining corresp - {}".format(x_X_new_image.shape))
        x_X_new_image = np.vstack((x_X_new_image, np.hstack((remaining_2d_3d, X_non_lin_tri))))
        print("(Nlinear)points after adding remaining corresp - {}".format(x_X_new_image.shape))

        # print reprojection error after non-linear triangulation
        pts_img_all = x_X_new_image[:, 0:2]
        X_all = x_X_new_image[:, 2:]
        reproj_errors = compute_reproj_err_all(pts_img_all, P_non_pnp, X_all)
        mean_proj_error[new_img_num].append(('NonLinTri', np.mean(reproj_errors)))

        # store the current pose after non linear pnp
        camera_poses_dict[new_img_num] = np.hstack((R_non_pnp, C_non_pnp.reshape((3, 1))))

        # plotting reprojected points
        # plot_funcs.plot_reproj_points(images[new_img_num-1], new_img_num, np.float32(pts_img_all), np.float32(pts_img_reproj_all), save=True)
        # print("plotting all the camera poses and their respective correspondences\n")
        x_X_map_dict[new_img_num] = np.hstack((pts_img_all, X_all))
        plot_funcs.plot_camera_poses(camera_poses_dict, x_X_map_dict, save=True)


        # do bundle adjustment
        index_start = X_set.shape[0]
        index_end = X_set.shape[0] + X_all.shape[0]
        x_Xindex_mapping[new_img_num] = zip(pts_img_all, range(index_start, index_end))
        X_set = np.append(X_set, X_all, axis=0)

        print("doing Bundle Adjustment --> ")
        pose_set_opt, X_set_opt = bundle_adjustment(camera_poses_dict, X_set, x_Xindex_mapping, K)
        print("keys --> {}".format(pose_set_opt.keys()))

        # compute reproj error after BA
        R_ba = pose_set_opt[new_img_num][:, 0:3]
        C_ba = pose_set_opt[new_img_num][:, 3]
        X_all_ba = X_set_opt[index_start:index_end].reshape((X_all.shape[0], 3))
        M_ba = misc_funcs.get_projection_matrix(K, R_ba, C_ba)
        reproj_errors = compute_reproj_err_all(pts_img_all, M_ba, X_all_ba)
        mean_proj_error[new_img_num].append(('BA', np.mean(reproj_errors)))


        # make X_set = X_set_opt and send it for further iterations ->
        X_set = X_set_opt

        # Save the optimal correspondences (2D-3D) and the optimal poses for next iteration
        x_X_map_dict[new_img_num] = np.hstack((pts_img_all, X_all_ba))
        # pose_set[new_img_num] = np.hstack((R_ba, C_ba.reshape((3, 1))))
        camera_poses_dict = pose_set_opt

        print("......................................")

    # plotting the output of BA
    plot_funcs.bundle_adjustment_op(pose_set_opt, X_set_opt)
    print(mean_proj_error)
if __name__ == "__main__":
    main()