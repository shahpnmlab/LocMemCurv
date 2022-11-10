import skimage
import scipy
import starfile
import mrcfile

import pylab as plt
import numpy as np

from pathlib import Path

from scipy.interpolate import UnivariateSpline
from scipy.ndimage import uniform_filter1d
from skimage.morphology import disk


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
    vol = np.swapaxes(mrcfile.read(volname), 2, 0)
    im_count = 0
    for coordinate in coordinates:
        im_count += 1
        box_x_start = int(coordinate[0] - dim_xy)
        box_x_end = int(coordinate[0] + dim_xy)
        box_y_start = int(coordinate[1] - dim_xy)
        box_y_end = int(coordinate[1] + dim_xy)
        box_z_start = int(coordinate[2] - dim_z)
        box_z_end = int(coordinate[2] + dim_z)

        cuboid = vol[box_x_start:box_x_end, box_y_start:box_y_end, box_z_start:box_z_end]
        img = np.sum(cuboid, axis=2)
        if not Path.is_dir(out_folder_name):
            Path.mkdir(out_folder_name, exist_ok=True)
            mrcfile.write(f"{out_folder_name}/pcle_{im_count}.mrc", img, overwrite=True)
        else:
            mrcfile.write(f"{out_folder_name}/pcle_{im_count}.mrc", img, overwrite=True)


def preprocess_img(img0: np.array, k1: float = 3, k2: float = 1.2) -> np.array:
    """
    preprocess the 2D image of the eDLP before calculating the radial intensity profiles
    :param img: np.array
    :param k1: parameter to control the kernel for the median filter
    :param k2: parameter to control the kernel for local contrast enhancement
    :return: filtered image
    """
    unscaled_img = mrcfile.read(img0)
    img = skimage.exposure.rescale_intensity(unscaled_img, out_range=(0, 255))
    img_m = skimage.filters.rank.median(img, footprint=disk(k1))
    img_c = skimage.filters.rank.enhance_contrast_percentile(img_m, footprint=disk(k2))
    sato = skimage.filters.sato(img_c, mode='wrap', black_ridges=False)
    return sato


def points_on_circle(radius, points, center):
    t = np.linspace(np.deg2rad(0), np.deg2rad(360), points)
    x = (radius * np.cos(t)) + center
    y = (radius * np.sin(t)) + center
    return np.vstack((x, y)).T


def average_intensity_profile(img: np.array, n_points: int, rad: int,
                              n_average: int, range_start: int,
                              range_end: int) -> np.array:
    """
    Calculates an average radial intensity profile of pixels of the underlying image emanating from the center of the
    box. profile can be truncated to perform peak detection only near the regions of interest by providing a start
    and end range.

    :param img: 2D img of the membrane enclosed DLP from the tomogram
    :param n_points: Number of points to sample on the circumference
    :param rad: Radius of the circle
    :param n_average: number of intensity profiles to average
    :param range_start: intensity profile start
    :param range_end: intensity profile end

    :return: averaged radial intensity profile
    """
    center = (img.shape[0] // 2) + 1
    circular_points = points_on_circle(rad, n_points, center)

    intensity_profile = np.empty((n_points, rad + 1))
    for i in np.arange(n_points):
        start = (center, center)
        end = np.array([circular_points[i][0], circular_points[i][1]])
        p = skimage.measure.profile_line(img, start, end)
        intensity_profile[i, :] = p[:rad + 1]
    #   ax[0].plot([start[0], end[0]],[start[1], end[1]], c='r')
    intensity_profile_avg = uniform_filter1d(intensity_profile, size=n_average, mode='wrap')
    intensity_profile_avg = intensity_profile_avg[:, range_start:range_end]
    return intensity_profile_avg, center


def local_curvature(coords, num_points, plot=False):
    coords_complete = np.vstack((coords, coords[1, :]))

    distance = np.cumsum(np.sqrt(np.sum(np.diff(coords_complete, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0) / distance[-1]

    splines = [UnivariateSpline(distance, coords, k=3) for coords in coords_complete.T]
    alpha = np.linspace(0, 1, num_points)
    points_fitted = np.vstack(spl(alpha) for spl in splines).T
    if plot:
        plt.plot(coords_complete[:, 0], coords_complete[:, 1], '-k', label='original')
        plt.plot(points_fitted[:, 0], points_fitted[:, 1], 'or', label='fitted')

        plt.axis('equal');
        plt.legend()
        plt.xlabel('x');
        plt.ylabel('y')
        plt.show()
    return coords_complete, points_fitted


star = "/Users/ps/data/wip/membeaneCurvature/tomo/run_data.star"
tsname = "Position_48"
volname = "/Users/ps/data/wip/membeaneCurvature/tomo/Position_48_10.00Apx.mrc"

k1 = 3
k2 = 1.5
rad = 60
n_points = 50
n_average = 10
d_alpha = np.pi * 2 / n_points
range_start = 37
range_end = 57
peak_height = 0.01
no_peak_count = 10

volumes = Path("tomo/").glob("*.mrc")

for volume in volumes:
    print("Taking 2D slices of particles")
    preprocess_tomo(star, tsname, volname, dim_xy=60, dim_z=7, out_folder_name=Path("wt"))

orig_imgs = Path("wt/").glob("*.mrc")

for orig_img in orig_imgs:

    d = preprocess_img(orig_img, k1, k2)
    # fig, ax = plt.subplots(1, 2)
    intensity_profile_avg, center = average_intensity_profile(d, n_points, n_average, range_start, range_end)

    def find_peaks(intensity_profile_avg, peak_height, range_start, center)
        global_radius = np.zeros(2)
        no_peaks = []

        for i in np.arange(n_points):
            peak_middle, _ = scipy.signal.find_peaks(intensity_profile_avg[i], height=peak_height)
            peak_outer, _ = scipy.signal.find_peaks(-np.gradient(intensity_profile_avg[i]), height=peak_height)
            peak_inner, _ = scipy.signal.find_peaks(np.gradient(intensity_profile_avg[i]), height=peak_height)

            peak_sum = 0
            peak_counter = 0
            if peak_middle.size != 0:
                peak_sum += peak_middle[-1]
                peak_counter += 1
            if peak_outer.size != 0:
                peak_sum += peak_outer[-1]
                peak_counter += 1
            if peak_inner.size != 0:
                peak_sum += peak_inner[-1]
                peak_counter += 1

            if peak_counter == 0:
                while len(no_peaks) <= no_peak_count:
                    no_peaks.append(i)
            else:
                peak = peak_sum / peak_counter
                plussing = np.array([peak, 1])
                global_radius += plussing
                peak = peak + range_start
            # ax[1].plot(-np.gradient(intensity_profile_avg[i]))

            # ax[1].plot(peak, -np.gradient(intensity_profile_avg[i]), 'kx')
            if peak_outer.size != 0:
                peak_outer = peak_outer[-1] + range_start
            if peak_middle.size != 0:
                peak_middle = peak_middle[-1] + range_start
            if peak_inner.size != 0:
                peak_inner = peak_inner[-1] + range_start

            xi_o = peak_outer * np.sin(i * d_alpha) + center
            yi_o = peak_outer * np.cos(i * d_alpha) + center
            xi_m = peak_middle * np.sin(i * d_alpha) + center
            yi_m = peak_middle * np.cos(i * d_alpha) + center
            xi_i = peak_inner * np.sin(i * d_alpha) + center
            yi_i = peak_inner * np.cos(i * d_alpha) + center
            detected_xi = peak * np.sin(i * d_alpha) + center
            detected_yi = peak * np.cos(i * d_alpha) + center
            global_radius[0] = global_radius[0] / global_radius[1]

            #for i in no_peaks:
            #    scale = (global_radius[0] + range_start) / rad

            #        ax[0].scatter(s*(circular_points[i][1]-center)+center, s*(circular_points[i][0]-center)+center, c='r', s=5)
            return detected_xi, detected_yi
        # ax[0].scatter(xi_o, yi_o, c='intensity_profile_avg', s=5)
        # ax[0].scatter(xi_m, yi_m, c='g', s=5)
        # ax[0].scatter(xi_i, yi_i, c='b', s=5)
        # ax[0].scatter(xi, yi, c='k', s=5)

    # ax[0].set_title(f'{orig_img.stem}')
    # ax[0].imshow(d, cmap='gray')
    # ax[1].set_title('Avg Profile')
    # plt.show()
