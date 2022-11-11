import numpy as np

def x_X_map(x_X_cur_image_map, xcur_xnew_map):
    
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