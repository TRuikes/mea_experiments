import os
import cv2
import numpy as np
import utils
import pandas as pd
from axorus.preprocessing.lib.filepaths import FilePaths
from axorus.preprocessing.laser_calibration.annotate_points_on_image import annotate_points
from axorus.preprocessing.project_colors import ProjectColors


# Function to calculate the center and radius of the circle
def calculate_circle_properties(binary_image):
    # Find contours of the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Use the largest contour to approximate a circle
        max_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(max_contour)
        return (int(x), int(y)), radius
    return None, 0


# Function to find boundaries for 90% of the volume
def find_boundaries_array(x, y, target_area=0.9):
    # Find the peak index
    peak_index = np.argmax(y)
    x_peak = x[peak_index]

    # Initialize indices for left and right boundaries
    left_index, right_index = peak_index, peak_index

    # Compute cumulative area starting from the peak
    current_area = 0.0
    total_area = np.trapz(y, x)

    while current_area / total_area < target_area:
        # Expand the indices symmetrically
        if left_index > 0:
            left_index -= 1
        if right_index < len(x) - 1:
            right_index += 1

        # Integrate over the selected range
        current_area = np.trapz(y[left_index:right_index + 1], x[left_index:right_index + 1])

    return x[left_index], x[right_index]


# Function to annotate extracted data to images
def annotate_image(im, cntr, r_pxl, d_um):
    # Write circle to image
    im_out = cv2.circle(im,
                        (int(cntr[0]), int(cntr[1])),
                        int(r_pxl),
                        (0, 255, 0),
                        2)

    im_out = cv2.putText(im_out,
                         f'Diameter: {d_um:.0f} um',
                         (int(cntr[0]) - 100, int(cntr[1] - 2 * r_pxl)),
                         cv2.FONT_HERSHEY_SIMPLEX,
                         1,
                         (0, 255, 0),
                         2
                         )
    return im_out


# Function to generate a circle of points based on a radius
def generate_circle_points(radius, num_points=100):
    """
    Generate x, y points for a circle with given radius.

    Parameters:
        radius (float): Radius of the circle.
        num_points (int): Number of points to generate along the circle.

    Returns:
        numpy.ndarray: Array of x coordinates.
        numpy.ndarray: Array of y coordinates.
    """
    # Generate angles from 0 to 2π
    theta = np.linspace(0, 2 * np.pi, num_points)

    # Calculate x and y coordinates
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    return x, y


# General setup
filepaths = FilePaths(laser_calib_week='week_48')

# Define the directory and threshold
directory = r"F:\Axorus\ex_vivo_series_3\laser_calibration\week_48\spotsizes\calibration_c7"

# Measure resolution of image
points = annotate_points(r"F:\Axorus\ex_vivo_series_3\laser_calibration\week_48\spotsizes\calibration_c7\mea.tif")
df = points.copy()

# Extract the x and y coordinates
coords = df[['x', 'y']].values

# Compute the pairwise distances
distances = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))

# Set the diagonal (distance to itself) to infinity to exclude self-comparison
np.fill_diagonal(distances, np.inf)

# Find the minimum distance for each point
nearest_distances = distances.min(axis=1)

mean_distance = np.mean(nearest_distances)

resolution_px_to_um = 30 / mean_distance  # um / pxl
# resolution_px_to_um = 2.10

#%% Read MEA image

# Load and analyze images
transections = {}
data_df = pd.DataFrame()
i = 0

for filename in os.listdir(directory):
    if not filename.startswith("z"):
        continue

    mea_image = cv2.imread(r"F:\Axorus\ex_vivo_series_3\laser_calibration\week_48\spotsizes\calibration_c7\mea.tif",
                           cv2.IMREAD_COLOR)

    filepath = os.path.join(directory, filename)

    # Read image
    im_spot_input = cv2.imread(filepath,
                               cv2.IMREAD_COLOR)

    # Convert to grayscale
    im_spot_gray = cv2.cvtColor(im_spot_input,
                                cv2.COLOR_BGR2GRAY)

    # Convert to binary
    if '10x' in filename:
        threshold = 150
    else:
        threshold = 100
    _, spot_mask = cv2.threshold(im_spot_gray,
                                 threshold,
                                 255,
                                 cv2.THRESH_BINARY)

    # Detect spot
    center_pxl, radius_pxl = calculate_circle_properties(spot_mask)

    if center_pxl is None:
        print(f'did not find circle for: {filename}')
        continue

    # convert mask to color
    spot_mask_clr = cv2.cvtColor(spot_mask, cv2.COLOR_GRAY2BGR)

    # Convert redius to um
    radius_um = radius_pxl * resolution_px_to_um
    diameter_um = 2 * radius_um

    # Write stats to output

    h, exp, gain = filename.split('.')[0].split('_')
    h = h[1:]

    k = (exp, gain)
    if k not in transections.keys():
        transections[k] = {}
    y = im_spot_gray[center_pxl[1], :].flatten()
    x = (np.arange(0, y.size) - center_pxl[0]) / resolution_px_to_um

    transections[k][h] = dict(
        x=x,
        y=y
    )

    data_df.at[i, 'z'] = int(h)
    data_df.at[i, 'exp'] = exp
    data_df.at[i, 'gain'] = gain
    data_df.at[i, 'radius_um'] = radius_um
    data_df.at[i, 'diameter_um'] = diameter_um

    i += 1

    # Print annotated mea image
    mea_annotated = annotate_image(mea_image, cntr=center_pxl, r_pxl=radius_pxl, d_um=diameter_um)

    # Print annotated spot image
    im_spot_input = annotate_image(im_spot_input, cntr=center_pxl, r_pxl=radius_pxl, d_um=diameter_um)

    # Crop images
    crop_size = 400  # um to keep
    crop_size_pxl = int(crop_size / resolution_px_to_um)

    mea_annotated = mea_annotated[center_pxl[1] - crop_size_pxl:center_pxl[1] + crop_size_pxl,
                    center_pxl[0] - crop_size_pxl:center_pxl[0] + crop_size_pxl,
                    :
                    ]
    im_spot_input = im_spot_input[center_pxl[1] - crop_size_pxl:center_pxl[1] + crop_size_pxl,
                    center_pxl[0] - crop_size_pxl:center_pxl[0] + crop_size_pxl,
                    :
                    ]

    # Concatenate images
    im_out = cv2.hconcat([mea_annotated, im_spot_input])
    savename = (filepaths.laser_calib_figure_dir / 'spotsize' / f'{gain}_{exp}_{h}_spot.png').as_posix()
    # Write image to disk
    cv2.imwrite(savename, im_out)
    print(f'saved: {savename}')

#%%

clrs = ProjectColors()

# Create figure
fig = utils.make_figure(
    equal_width_height='y',
    # equal_width_height_axes=[[1, 2]],
    width=1.2, height=1.4,
    x_domains={
        1: [[0.05, 0.25], [0.4, 0.6], [0.7, 0.9]],
        2: [[0.05, 0.25], [0.4, 0.6], [0.7, 0.9]],

    },
    y_domains={
        1: [[0.6, 0.95] for _ in range(3)],
        2: [[0.05, 0.45] for _ in range(3)],
    },

    subplot_titles={
        1: ['short exposure', '', ''],
        2: ['long exposure', '', '']
    }
)

for dtype in [('80ms', '1x'), ('100ms', '10x')]:
    if dtype == ('80ms', '1x'):
        row = 1
    else:
        row = 2

    pos = dict(row=row, col=1)
    for k in transections[dtype].keys():
        x = transections[dtype][k]['x']
        y = transections[dtype][k]['y']
        fig.add_scatter(
            x=x,
            y=y,
            showlegend=False,
            line=dict(color=clrs.min_max_map(int(k), 0, 1001), width=0.5),
            **pos,
        )

    fig.update_xaxes(
        tickvals=np.arange(-500, 500, 25),
        range=[-100, 100],
        title_text='x [um]',
        **pos,
    )

    pos = dict(row=row, col=2)

    fig.add_scatter(
        x=np.arange(-100, 100, 10),
        y=np.arange(-100, 100, 10),
        line=dict(color='white'),
        showlegend=False,
        mode='lines',
        **pos,
    )

    d = data_df.query(f'exp == "{dtype[0]}" and gain == "{dtype[1]}"')
    for i, r in d.iterrows():
        x, y = generate_circle_points(r.radius_um, num_points=100)

        fig.add_scatter(
            x=x, y=y, mode='lines',
            line=dict(color=clrs.min_max_map(int(r.z), 0, 1001), width=0.5),
            showlegend=False,
            **pos,
        )

    fig.update_xaxes(
        tickvals=np.arange(-80, 100, 20),
        range=[-80, 80],
        title_text='x [um]',
        **pos,
    )

    fig.update_yaxes(
        tickvals=np.arange(-80, 100, 20),
        range=[-80, 80],
        title_text='y [um]',
        **pos,
    )

    pos = dict(row=1, col=3)

    d.sort_values('z', inplace=True)

    x = d.z.values
    y = d.diameter_um.values
    fig.add_scatter(
        x=x, y=y,
        mode='lines', line=dict(color='blue' if row == 1 else 'green', width=1),
        showlegend=False,
        **pos,
    )

    fig.update_xaxes(
        tickvals=np.arange(0, 1000, 200),
        title_text=f'Z [um]',
        **pos,
    )

    fig.update_yaxes(
        tickvals=[0, 25, 50, 75, 100, 150, 200],
        title_text=f'Spot diameter [um]',
        **pos,
    )


utils.save_fig(fig, filepaths.laser_calib_figure_dir / 'spotsize' / 'summary', display=True)
