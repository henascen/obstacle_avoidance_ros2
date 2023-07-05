def bresenham_points(p0, p1):
    # Taken from the example

    points_list = []

    x0, y0 = p0[0], p0[1]
    x1, y1 = p1[0], p1[1]

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    if x0 < x1:
        sx = 1
    else:
        sx = -1
    
    if y0 < y1:
        sy = 1
    else:
        sy = -1
    
    err = dx - dy

    while True:
        points_list.append([x0, y0])

        if x0 == x1 and y0 == y1:
            break
        
        e2 = 2*err
        if e2 > -dy:
            # overshot in the y direction
            err = err - dy
            x0 = x0 + sx
        if e2 < dx:
            # overshot in the x direction
            err = err + dx
            y0 = y0 + sy
    
    del points_list[0]
    del points_list[-1]

    return points_list