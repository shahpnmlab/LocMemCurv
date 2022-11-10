import skimage
import scipy
import starfile
import mrcfile

import pylab as plt
import numpy as np

from pathlib import Path
from typing import Literal
from scipy.ndimage import uniform_filter1d
from skimage.morphology import disk

def imshow(i, disp:Literal[True, False]=False):
    plt.imshow(i, cmap='gray')
    plt.axis('equal')
    if disp == True:
        plt.show()

def show(i, style:Literal['scatter', 'plot']='scatter', disp:Literal[True, False]=False):
    if style == 'scatter':
        plt.scatter(i[:,0], i[:,1])
    if style == 'plot':
        plt.plot(i)
    plt.axis('equal')
    if disp == True:
        plt.show()

def overlay(data, points):
    fig, ax = plt.subplots()
    ax.imshow(data,cmap=plt.cm.gray)
    ax.scatter(points[:,0], points[:,1])
    plt.show()

def points_on_circle(radius, points, center):
    t = np.linspace(np.deg2rad(0),np.deg2rad(360), points)
    x = (radius * np.cos(t)) + center
    y = (radius * np.sin(t)) + center
    return np.vstack((x,y)).T

def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center)**2 + (y - center)**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile


def read_starfile(file_name, tomo_name):
    star = starfile.read(file_name)
    pixel_size = star['optics']['rlnImagePixelSize'].to_numpy()
    data = star['particles'].merge(star['optics'])
    data = data[data['rlnMicrographName'].str.contains(tomo_name)]
    xyz_headings = [f'rlnCoordinate{axis}' for axis in 'XYZ']
    shift_headings = [f'rlnOrigin{axis}Angst' for axis in 'XYZ']
    xyz = data[xyz_headings].to_numpy()
    shifts_ang = data[shift_headings].to_numpy()
    shifts_px = shifts_ang / pixel_size
    xyz_shifted = xyz - shifts_px
    return xyz_shifted


def preprocess_tomo(star, tsname, volname, dim_xy=52, dim_z=7, out_folder_name="2dslices"):
    coordinates = read_starfile(star, tsname)
    vol = np.swapaxes(mrcfile.read(volname), 2,0)
    im_count=0
    for coordinate in coordinates:
        im_count += 1
        box_x_start = int(coordinate[0] - dim_xy)
        box_x_end = int(coordinate[0] + dim_xy)
        box_y_start = int(coordinate[1] - dim_xy)
        box_y_end = int(coordinate[1] + dim_xy)
        box_z_start = int(coordinate[2] - dim_z)
        box_z_end = int(coordinate[2] + dim_z)

        cuboid = vol[box_x_start:box_x_end,box_y_start:box_y_end,box_z_start:box_z_end]
        img = np.sum(cuboid, axis=2)
        if not Path.is_dir(out_folder_name):
            Path.mkdir(out_folder_name, exist_ok=True)
            mrcfile.write(f"{out_folder_name}/pcle_{im_count}.mrc", img, overwrite=True)
        else:
            mrcfile.write(f"{out_folder_name}/pcle_{im_count}.mrc", img, overwrite=True)

star = "/Users/ps/data/wip/membeaneCurvature/tomo/run_data.star"
tsname = "Position_48"
volname = "/Users/ps/data/wip/membeaneCurvature/tomo/Position_48_10.00Apx.mrc"


k1 = 3
k2 = 1.5
rad = 60
n_points = 50
n_average = 5
d_alpha = np.pi * 2 / n_points
range_start = 37
range_end = 50
peak_height = 0.005
no_peak_count = 5


volumes = Path("tomo/").glob("*.mrc")

for volume in volumes:
    print("Taking 2D slices of particles")
    preprocess_tomo(star, tsname, volname,dim_xy=60, dim_z=7, out_folder_name=Path("wt"))

orig_imgs = Path("wt/").glob("*.mrc")
for orig_img in orig_imgs:
    unscaled_img = mrcfile.read(orig_img)
    img = skimage.exposure.rescale_intensity(unscaled_img, out_range=(0,1))
    center = (img.shape[0] // 2) + 1
    img_m = skimage.filters.rank.median(img, footprint=disk(k1))
    img_c = skimage.filters.rank.enhance_contrast_percentile(img_m, footprint=disk(k2))
    #sato = skimage.filters.sato(img_m, mode='wrap', black_ridges=False)
    circular_points = points_on_circle(rad, n_points, center)
    fig, ax = plt.subplots(1, 2)
    foo = np.empty((n_points, rad+1))
    d = img_m
    for i in np.arange(n_points):
        start = (center, center)
        end0 = np.array([circular_points[i][0], circular_points[i][1]])
        p0 = skimage.measure.profile_line(d, start, end0)
        foo[i,:] = p0[:rad+1]
    #        ax[0].plot([start[0], end0[0]],[start[1], end0[1]], c='r')

    y = uniform_filter1d(foo, size=n_average, mode='wrap')
    y = y[:,range_start:range_end]

    global_radius = np.zeros(2)
    no_peaks = []
    for i in np.arange(n_points):
        peak, _ = scipy.signal.find_peaks(y[i], height=peak_height)
        if peak.size == 0:
            while len(no_peaks) <= no_peak_count:
                no_peaks.append(i)
        else:
            peak = peak[-1]
            # ax[0].scatter( circular_points[i][0], circular_points[i][1], c='b')
            plussing = np.array([peak, 1])
            global_radius += plussing
        ax[1].plot(-np.gradient(y[i]))

        #ax[1].plot(peak, -np.gradient(y[i]), 'kx')

        peak = peak + range_start

        xi = peak * np.sin(i * d_alpha) + center
        yi = peak * np.cos(i * d_alpha) + center

        ax[0].scatter(xi, yi, c='g', s=15)
    global_radius[0] = global_radius[0] / global_radius[1]

    for i in no_peaks:
        s = (global_radius[0]+range_start) / rad
        ax[0].scatter(s*(circular_points[i][1]-center)+center, s*(circular_points[i][0]-center)+center, c='r', s=5)
    ax[0].set_title('Image')
    ax[0].imshow(d, cmap='gray')
    ax[1].set_title('Avg Profile')
    plt.show()

