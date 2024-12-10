import os
import cv2
import numpy as np
import plotly.graph_objects as go
import utils
import pandas as pd
from axorus.preprocessing.lib.filepaths import FilePaths
from axorus.preprocessing.laser_calibration.annotate_points_on_image import annotate_points
from axorus.preprocessing.project_colors import ProjectColors

filepaths = FilePaths(laser_calib_week='week_48')

# Define the directory and threshold
directory = r"F:\Axorus\ex_vivo_series_3\laser_calibration\week_48\spotsizes\calibration_c7"
threshold_value = 50  # User-defined threshold

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

print(f'Resolution = {resolution_px_to_um:3f} um / pxl')




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



# Load and analyze images
results = {}
transections = {}
data_df = pd.DataFrame()
i = 0
for filename in os.listdir(directory):
    if filename.startswith("z") and filename.endswith(".tif"):
        filepath = os.path.join(directory, filename)
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
        center, radius = calculate_circle_properties(binary_image)
        results[filename] = {"image": image, "center": center, "radius": radius}

        h, exp, gain = filename.split('.')[0].split('_')
        h = h[1:]

        k = (exp, gain)
        if k not in transections.keys():
            transections[k] = {}
        y = image[center[1], :].flatten()
        x = (np.arange(0, y.size) - center[0]) / resolution_px_to_um

        transections[k][h] = dict(
            x=x,
            y=y
        )

        y = image[center[1], :].flatten()
        x = (np.arange(0, y.size) - center[0]) / resolution_px_to_um

        data_df.at[i, 'z'] = h
        data_df.at[i, 'exp'] = exp
        data_df.at[i, 'gain'] = gain
        data_df.at[i, 'radius'] = radius

        i += 1

# Generate plots for images
figures = []
for filename, data in results.items():
    image = data["image"]
    center = data["center"]
    radius = data["radius"]

    h, exp, gain = filename.split('.')[0].split('_')
    h = h[1:]

    # Create figure
    fig = utils.make_figure(
        equal_width_height='y',
        width=1.2, height=1,
        x_domains={
            1: [[0.1, 0.4], [0.6, 0.9]],

        },
        y_domains={
            1: [[0.1, 0.9] for _ in range(2)]
        },
        subplot_titles={
            1: [f'Z={h} (gain={gain}, exposure={exp})', '']
        }

    )

    # Plot image + annotations
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Plot image
    fig.add_trace(go.Image(z=image_rgb), row=1, col=1)

    # Add a circle overlay
    fig.add_shape(
        type="circle",
        x0=center[0] - radius,
        y0=center[1] - radius,
        x1=center[0] + radius,
        y1=center[1] + radius,
        line=dict(color="red", width=1),
        row=1, col=1,
        showlegend=False,
    )

    # Add line through circle centre overlay
    fig.add_shape(
        type="line",
        x0=0,
        y0=center[1],
        x1=image.shape[1],
        y1=center[1],
        line=dict(color="red", width=1, dash="2px"),
        row=1, col=1, showlegend=False,
    )

    # Annotate dimensions of detected circle
    fig.add_annotation(
        x=center[0],
        y=center[1],
        text=f"Radius: {radius / resolution_px_to_um:.0f} um",
        font=dict(color='red', size=6),
        row=1, col=1
    )

    fig.update_xaxes(
        tickvals=np.arange(0, 1400, 200),
        ticktext=[f'{f:.0f}' for f in np.arange(0, 1400, 200) / resolution_px_to_um],
        title_text='x [um]',
        row=1, col=1,
    )

    fig.update_yaxes(
        tickvals=np.arange(0, 1400, 200),
        ticktext=[f'{f:.0f}' for f in np.arange(0, 1400, 200) / resolution_px_to_um],
        title_text='y [um]',
        row=1, col=1,
    )

    # Plot the intensity profile through the enter of the circle
    pos = dict(row=1, col=2)

    y = image[center[1], :].flatten()
    x = (np.arange(0, y.size) - center[0]) / resolution_px_to_um

    fig.add_scatter(
        x=x, y=y, mode='lines', line=dict(color='black', width=1),
        showlegend=False,
        **pos,
    )
    fig.add_scatter(
        x=[x[0], x[-1]], y=[threshold_value, threshold_value],
        mode='lines', line=dict(color='red', width=1),
        showlegend=False,
        **pos,
    )

    fig.update_xaxes(
        tickvals=np.arange(-300, 400, 100),
        title_text='x [um]',
        **pos,
    )

    fig.update_yaxes(
        tickvals=np.arange(0, 300, 50),
        **pos,
    )

    utils.save_fig(fig, filepaths.laser_calib_figure_dir / 'spotsize' / f'{gain}_{exp}_{h}')

#%%

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


#%% tmp


filename = list(results.keys())[2]
data = results[filename]
image = data["image"]
center = data["center"]
radius = data["radius"]

h, exp, gain = filename.split('.')[0].split('_')
h = h[1:]

# Create figure
fig = utils.make_figure(
    equal_width_height='y',
    width=1.2, height=1,
    x_domains={
        1: [[0.1, 0.4], [0.6, 0.9]],

    },
    y_domains={
        1: [[0.1, 0.9] for _ in range(2)]
    },
    subplot_titles={
        1: [f'Z={h} (gain={gain}, exposure={exp})', '']
    }

)

image = cv2.imread(r"F:\Axorus\ex_vivo_series_3\laser_calibration\week_48\spotsizes\calibration_c7\mea.tif", cv2.IMREAD_GRAYSCALE)
image_spot = cv2.imread(r"F:\Axorus\ex_vivo_series_3\laser_calibration\week_48\spotsizes\calibration_c7\z0_80ms_1x.tif", cv2.IMREAD_GRAYSCALE)

# Plot image + annotations
image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
image_rgb_spot = cv2.cvtColor(image_spot, cv2.COLOR_GRAY2RGB)

# Plot image
fig.add_trace(go.Image(z=image_rgb), row=1, col=1)

# Add a circle overlay
fig.add_shape(
    type="circle",
    x0=center[0] - radius,
    y0=center[1] - radius,
    x1=center[0] + radius,
    y1=center[1] + radius,
    line=dict(color="red", width=1),
    row=1, col=1,
    showlegend=False,
)

# Add line through circle centre overlay
fig.add_shape(
    type="line",
    x0=0,
    y0=center[1],
    x1=image.shape[1],
    y1=center[1],
    line=dict(color="red", width=1, dash="2px"),
    row=1, col=1, showlegend=False,
)

# Annotate dimensions of detected circle
fig.add_annotation(
    x=center[0],
    y=center[1],
    text=f"Radius: {radius / resolution_px_to_um:.0f} um",
    font=dict(color='red', size=6),
    row=1, col=1
)

fig.update_xaxes(
    tickvals=np.arange(0, 1400, 200),
    ticktext=[f'{f:.0f}' for f in np.arange(0, 1400, 200) / resolution_px_to_um],
    title_text='x [um]',
    row=1, col=1,
)

fig.update_yaxes(
    tickvals=np.arange(0, 1400, 200),
    ticktext=[f'{f:.0f}' for f in np.arange(0, 1400, 200) / resolution_px_to_um],
    title_text='y [um]',
    row=1, col=1,
)


# Plot image
fig.add_trace(go.Image(z=image_rgb_spot), row=1, col=2)

# Add a circle overlay
fig.add_shape(
    type="circle",
    x0=center[0] - radius,
    y0=center[1] - radius,
    x1=center[0] + radius,
    y1=center[1] + radius,
    line=dict(color="red", width=1),
    row=1, col=2,
    showlegend=False,
)

# Add line through circle centre overlay
fig.add_shape(
    type="line",
    x0=0,
    y0=center[1],
    x1=image.shape[1],
    y1=center[1],
    line=dict(color="red", width=1, dash="2px"),
    row=1, col=2, showlegend=False,
)

# Annotate dimensions of detected circle
fig.add_annotation(
    x=center[0],
    y=center[1],
    text=f"Radius: {radius / resolution_px_to_um:.0f} um",
    font=dict(color='red', size=6),
    row=1, col=2
)

fig.update_xaxes(
    tickvals=np.arange(0, 1400, 200),
    ticktext=[f'{f:.0f}' for f in np.arange(0, 1400, 200) / resolution_px_to_um],
    title_text='x [um]',
    row=1, col=2,
)

fig.update_yaxes(
    tickvals=np.arange(0, 1400, 200),
    ticktext=[f'{f:.0f}' for f in np.arange(0, 1400, 200) / resolution_px_to_um],
    title_text='y [um]',
    row=1, col=2,
)

utils.save_fig(fig, filepaths.laser_calib_figure_dir / 'spotsize' / 'spotsize_on_mea')

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
        tickvals=np.arange(-500, 500, 100),
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
        x, y = generate_circle_points(r.radius, num_points=100)

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
    dp = d.copy()
    dp['z_int'] = dp.z.astype('int')
    dp.sort_values('z_int', inplace=True)

    x = dp.z.astype('int').values
    y = dp.radius.values * 2
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
