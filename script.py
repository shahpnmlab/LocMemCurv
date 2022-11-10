import numpy as np
import pylab as plt
import imodmodel

from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline

def read_mod(modfile, sampling=10):
    df = imodmodel.read(modfile)
    coords_and_contours = df[['x','y','contour_id']]
    unique_contours = df['contour_id'].unique()
    return coords_and_contours, unique_contours

def get_coords(all_coords, con_id):
        points_in_contour_df = all_coords.where(all_coords['contour_id'] == con_id)
        points_in_contour = points_in_contour_df.dropna().to_numpy()[:,0:2]
        return points_in_contour

def define_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return (None, np.inf)

    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((cx,cy), radius)

def interpolate_coords(coords, num_points, plot=False):
    coords_complete =  np.vstack((coords, coords[1,:]))

    #x = coords_complete[:,0]
    #y = coords_complete[:,1]
    #xd = np.diff(x)
    #yd = np.diff(y)
    #dist = np.sqrt(xd**2, yd**2)
    distance = np.cumsum( np.sqrt(np.sum( np.diff(coords_complete, axis=0)**2, axis=1 )) )
    distance = np.insert(distance, 0, 0)/distance[-1]
    #u = np.cumsum(dist)
    #u = np.hstack([[0],u])
    
    #t = np.linspace(0,u.max(),coords_complete.shape[0])
    #xn = np.interp(t, u, x)
    #yn = np.interp(t, u, y)
    #splines = [UnivariateSpline(distance, coords, k=3, s=0.2) for coords in coords_complete.T]
    splines = [UnivariateSpline(distance, coords, k=3) for coords in coords_complete.T]
    alpha = np.linspace(0, 1, num_points)
    points_fitted = np.vstack( spl(alpha) for spl in splines ).T
    if plot:
        plt.plot(O[:,0],O[:,1], '-k', label='original')
        plt.plot(X[:,0],X[:,1], 'or', label='fitted')

        plt.axis('equal'); plt.legend() 
        plt.xlabel('x'); plt.ylabel('y')
        plt.show()
    return coords_complete, points_fitted





#    out = np.empty(coords.shape[0])
#    for i in np.arange(coords_complete.shape[0]-2)+1:
#        center, radius = define_circle(coords_complete[i-1], coords_complete[i], coords_complete[i+1])
#        out[i-1] = 1/radius



all_points, conts = read_mod("funny.mod")
coords = get_coords(all_points, 0)
O, X = interpolate_coords(coords, 250, plot=False)

out = np.empty(X.shape[0])
for i in np.arange(X.shape[0]-2)+1:
    center, radius = define_circle(X[i-1], X[i], X[i+1])
    out[i-1] = 1/radius

ax1 = plt.subplot(121)
ax1.plot(O[1:-1,0], O[1:-1,1],c='k')
ax1.scatter(X[1:-1,0], X[1:-1,1], s=out[:-2]*1000)
plt.axis('equal')
plt.xlabel('x'); plt.ylabel('y')
ax2 = plt.subplot(122, sharex=ax1)
ax2.plot(O[1:-1,0], O[1:-1,1],c='k')
ax2.scatter(X[1:-1,0], X[1:-1,1], c=out[:-2], cmap='jet')
plt.axis('equal')
plt.xlabel('x'); plt.ylabel('y')
plt.show()
