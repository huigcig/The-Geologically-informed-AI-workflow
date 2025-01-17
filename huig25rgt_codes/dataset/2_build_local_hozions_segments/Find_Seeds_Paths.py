# huig25rgt 2024.11.5 #
import random
import numpy as np
import matplotlib.pyplot as plt

def pickLocalPath(local_target2, center, method="1x1"):
    """
    Optimized flood fill algorithm using BFS
    """
    height, width = local_target2.shape
    local_path = np.zeros_like(local_target2)
    # Define directions based on method
    if method == "2x2":
        directions = np.array([[-2, 2], [-1, 2], [0, 2], [1, 2], [2, 2],
                               [-2, 1], [-1, 1], [0, 1], [1, 1], [2, 1],
                               [-2, 0], [-1, 0], [1, 0], [2, 0],
                               [-2, -1], [-1, -1], [0, -1], [1, -1], [2, -1],
                               [-2, -2], [-1, -2], [0, -2], [1, -2], [2, -2]])
    elif method == "1x1":
        directions = np.array([[-1, 1], [0, 1], [1, 1],
                               [-1, 0],         [1, 0],
                               [-1, -1], [0, -1], [1, -1]])
    elif method == "flood":
        directions = np.array([[0, 1], [-1, 0], [1, 0], [0, -1]])
    else:
        raise ValueError("Invalid method. Method should be '1x1', '2x2', or 'flood'.")

    # Initialize BFS queue with the center point
    queue = [(center[0], center[1])]
    local_coord = []
    while queue:
        x, y = queue.pop(0)  # Dequeue the first element
        if local_path[x, y] == 0:  # Check if not visited
            local_path[x, y] = 1
            local_coord.append((x, y))

            # Compute neighbor coordinates
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < height and 0 <= ny < width and local_target2[nx, ny] == 1:
                    if local_path[nx, ny] == 0:
                        queue.append((nx, ny))  # Enqueue new valid point

    return local_path, local_coord

def pick_seeds_paths(fx,li,sl,width=10,fill_method="1x1"):
    """
    Picking all seeds and local paths defined by Hui Gao (2024.10.23)
    Parameters:
    fx: cigfacies data
    li: linearity
    width: windows' half-width
    fill_method: the method of fill-flood method using 1x1 window or 2x2 window
    """
    # find all seeds points and linearity value
    h,w = fx.shape
    seeds_val = np.max(fx)
    si = np.argwhere(fx == seeds_val)
    si_li = li[fx == seeds_val]
    si_sl = sl[fx == seeds_val]
    # sort the points based on linearity or slope
    sort_rule = np.argsort(si_li)[::-1] # coord
#     sort_rule = np.argsort(si_sl)[::-1] # coord
    sort_si = si[sort_rule] # sorted points num*[x,y]
    sort_li = si_li[sort_rule] # sorted linearity value 
    sort_sl = si_sl[sort_rule] # sorted slope value 
    
    PATH = []
    # pick the local path 
    while len(sort_si)!=0 :
        xi,yi = sort_si[0]
        x1,x2 = max(0,xi-width),min(h,xi+width+1) # local windows
        y1,y2 = max(0,yi-width),min(w,yi+width+1)
        center = [int(xi-x1),int(yi-y1)]
        local_fx = fx[x1:x2,y1:y2].copy()
        local_path, local_seeds = pickLocalPath(local_fx,center,fill_method) # local path with center

        # add the local size
        local_path_resize = np.zeros((2*width+1,2*width+1))
        local_path_resize[0:int(x2-x1),0:int(y2-y1)] = local_path
        # if the path are too small, don't save the local path
        if np.sum(local_path) > 10: 
            update_seeds = np.array(local_seeds) + [x1,y1]
            PATH.append(([xi,yi],local_path_resize, update_seeds))

        # remove the path points from seeds 
        if sort_sl[0] > 0.17:
            x01,x02 = int((xi+x1)/2-x1),int((xi+x2)/2-x1)
            y01,y02 = int((yi+y1)/2-y1),int((yi+y2)/2-y1)
            for xii,yii in local_seeds:
                if (xii<x01 or xii>x02) and (yii<y01 or yii>y02):
                    local_seeds.remove((xii,yii))
        remove_seeds = np.array(local_seeds) + [x1,y1] # transfer to fx coord
        remove_coord = ~np.any(np.all(sort_si[:,None] == remove_seeds, axis=2), axis=1)
        sort_si = sort_si[remove_coord] # Updated seeds
        sort_li = sort_li[remove_coord] # Updated linearity value 

    return PATH