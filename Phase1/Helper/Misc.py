import numpy as np
import matplotlib.pyplot as plt

def x_X_map_creator(x_X_cur_image_map, xcur_xnew_map):
    
    #initializing
    x_X_new_img_map = []
    left_over_xcur_xnew = []

    for a_point in xcur_xnew_map:
        x_cur = a_point[0:2]
        x_new = a_point[2:4]

        # compare with all other points in ref image to get
        x_list_cur_image = x_X_cur_image_map[:, 0:2]
        error = x_list_cur_image - x_cur
        error = error**2
        error = np.sum(error, axis=1)
        error = np.sqrt(error)
        locs = np.where(error < 1e-3)[0]

        if(np.shape(locs)[0] == 1):
            x_X_new_img_map.append([x_new[0], x_new[1],x_X_cur_image_map[locs][0][2], x_X_cur_image_map[locs][0][3], x_X_cur_image_map[locs][0][4]])
        else:
            left_over_xcur_xnew.append(a_point)

    return np.array(x_X_new_img_map), np.array(left_over_xcur_xnew)

def plot_3D(X_list,camera_locations,name="test"):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    

    camera_locations=camera_locations.ravel()
    ax.scatter(camera_locations[0],camera_locations[1],camera_locations[2],color='red',marker='x')

    x=X_list[:,0]
    y=X_list[:,1]
    z=X_list[:,2]
    ax.scatter(x.ravel(), y.ravel(), -z.ravel(), marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title("name")
    plt.show()

def plot_after_BA(poses_set,X_set):
    c_list=[]
    for a_pose in poses_set:
        C=poses_set[a_pose][:,-1]
        C=C.reshape(-1)
        c_list.append(C)
    c_list=np.array(c_list)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')


    cx,cy,cz=c_list[:,0],c_list[:,1],c_list[:,2]
    ax.scatter(cx.ravel(), cy.ravel(), -cz.ravel(), color='red',marker='x')
    x=X_set[:,0]
    y=X_set[:,1]
    z=X_set[:,2]
    ax.scatter(x.ravel(), y.ravel(), -z.ravel(), marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title("name")
    plt.show()
