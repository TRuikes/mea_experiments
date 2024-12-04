import cv2
import pandas as pd

# Initialize variables
points = []  # Store points as (x, y) relative to the original image
rect_start = None
rect_end = None
zoomed_image = None
original_image = None
image = None
scale_x, scale_y = 1, 1  # Scaling factors for x and y axes
offset_x, offset_y = 0, 0  # Offset for cropped region


# Function to load an image using OpenCV
def load_image(image_file):
    # Ask user to select an image file (file path)
    image = cv2.imread(image_file)  # You could adjust this for file dialog
    return image


# Function to handle mouse events
def on_mouse(event, x, y, flags, param):
    global points, rect_start, rect_end, zoomed_image, image, original_image, scale_x, scale_y, offset_x, offset_y

    if event == cv2.EVENT_RBUTTONDOWN:  # Right click to add a point
        # Convert screen coordinates back to original image coordinates
        original_x = int((x / scale_x) + offset_x)
        original_y = int((y / scale_y) + offset_y)
        points.append((original_x, original_y))  # Store relative to original
        draw_points(image)
        cv2.imshow("Image", image)

    elif event == cv2.EVENT_LBUTTONDOWN:  # Left click to start rectangle
        rect_start = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and rect_start is not None:  # Dragging rectangle
        rect_end = (x, y)
        temp_image = image.copy()
        cv2.rectangle(temp_image, rect_start, rect_end, (0, 255, 0), 2)
        draw_points(temp_image)  # Keep points visible while drawing
        cv2.imshow("Image", temp_image)

    elif event == cv2.EVENT_LBUTTONUP:  # Release left click to zoom
        if rect_start and rect_end:
            x1, y1 = rect_start
            x2, y2 = rect_end

            # Ensure coordinates are valid
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])

            # Update offsets and scaling factors
            offset_x, offset_y = int(x1 / scale_x) + offset_x, int(y1 / scale_y) + offset_y
            width = int((x2 - x1) / scale_x)
            height = int((y2 - y1) / scale_y)

            # Crop and zoom
            zoomed_image = original_image[offset_y:offset_y + height, offset_x:offset_x + width]
            scale_x = original_image.shape[1] / zoomed_image.shape[1]
            scale_y = original_image.shape[0] / zoomed_image.shape[0]
            zoomed_image = cv2.resize(zoomed_image, (original_image.shape[1], original_image.shape[0]))
            image = zoomed_image.copy()
            rect_start, rect_end = None, None
            draw_points(image)  # Restore points on the zoomed image
            cv2.imshow("Image", image)


# Function to draw points on the image
def draw_points(img):
    for px, py in points:
        # Scale the original coordinates to the current view
        display_x = int((px - offset_x) * scale_x)
        display_y = int((py - offset_y) * scale_y)
        cv2.circle(img, (display_x, display_y), 5, (0, 0, 255), -1)


# Main program
def annotate_points(image_file):
    global original_image, image, zoomed_image, scale_x, scale_y, offset_x, offset_y

    # Load image
    original_image = load_image(image_file)
    image = original_image.copy()

    # Reset scaling and offset
    scale_x, scale_y = 1, 1
    offset_x, offset_y = 0, 0

    # Set up the OpenCV window and mouse callback
    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", on_mouse)

    # Main loop
    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # Quit
            break
        elif key == ord('r'):  # Reset
            image = original_image.copy()
            zoomed_image = None
            scale_x, scale_y = 1, 1
            offset_x, offset_y = 0, 0
            draw_points(image)  # Preserve points
            cv2.imshow("Image", image)

    # Close OpenCV windows
    cv2.destroyAllWindows()

    return return_points()


# Save points to a CSV file
def return_points():
    df = pd.DataFrame(points, columns=["x", "y"])
    file_name = "points.csv"
    df.to_csv(file_name, index=False)
    print(f"Points saved to {file_name}")
    return df

