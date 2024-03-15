import argparse
import os
import threading
import time

import cv2
import numpy as np
from decord import VideoReader
from decord import cpu

# Initialize a lock for thread-safe operations on the video capture object
video_capture_lock = threading.Lock()

# Set the desired resolution
FRAME_WIDTH = 854  # 256
FRAME_HEIGHT = 480  # 144

# Time interval (in seconds) between extracted frames
DEFAULT_EXTRACT_INTERVAL = 2

# Threshold value for template matching
MATCH_THRESH = 0.85

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))


def load_and_resize_image(image_path, scale_percent=50):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Calculate the new dimensions
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # Resize the image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


template_images = {
    "double": load_and_resize_image(os.path.join(current_dir, 'data', 'images of phrases', 'double.png'), 100),
    "triple": load_and_resize_image(os.path.join(current_dir, 'data', 'images of phrases', 'triple.png'), 100),
    "quadra": load_and_resize_image(os.path.join(current_dir, 'data', 'images of phrases', 'quadra.png'), 100),
    "penta": load_and_resize_image(os.path.join(current_dir, 'data', 'images of phrases', 'penta.png'), 100),
}


# for name, image in template_images.items():
#     cv2.imshow(name, image)
#     cv2.waitKey(0) # Wait for a key press to move to the next image
# cv2.destroyAllWindows() # Close all OpenCV windows


def extract_frames(video_file_path, seconds_between_frames):
    vr = VideoReader(video_file_path, ctx=cpu(0))  # Use GPU if available
    fps = vr.get_avg_fps()
    frames_to_extract = int(fps * seconds_between_frames)

    total_frames = len(vr)
    frame_indices = list(range(0, total_frames, frames_to_extract))

    extracted_frames = []
    for idx in frame_indices:
        frame = vr[idx].asnumpy()
        timestamp = idx / fps

        # Resize frame to lower resolution (144p) for faster processing
        frame_resized = frame  # cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        classified_frame = classify_frame(gray_frame, timestamp)
        if classified_frame:
            extracted_frames.append(classified_frame)

        # # Display the processed frame
        # cv2.imshow(f'Frame {idx}', gray_frame)
        # cv2.waitKey(0) # Wait for a key press to move to the next image
        # cv2.destroyAllWindows()
    return extracted_frames


def show_frame_with_border(frame, key):
    """
    Show the frame with a border around the matched template.

    Args:
        frame (numpy.array): Grayscale image frame.
        key (str): Key indicating the matched template.
    """
    result = cv2.matchTemplate(frame, template_images[key], cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= MATCH_THRESH)

    for pt in zip(*loc[::-1]):
        bottom_right = (pt[0] + template_images[key].shape[1], pt[1] + template_images[key].shape[0])
        cv2.rectangle(frame, pt, bottom_right, (0, 0, 255), 2)

    frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    cv2.imshow('Frame with Border', frame)
    cv2.waitKey(0)  # Wait until a key is pressed (0 means infinite)
    cv2.destroyAllWindows()


def classify_frame(frame, timestamp):
    """
    Classify a single frame using template matching.

    Args:
        frame (numpy.array): Grayscale image frame.
        timestamp (float): Timestamp of the frame in seconds.

    Returns:
        str: String indicating the matched template and its timestamp.
    """
    for key in template_images:
        result = cv2.matchTemplate(frame, template_images[key], cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= MATCH_THRESH)

        # Output True if there is a match, False otherwise
        if len(loc[0]) > 0:
            if SHOW_MATCH:
                print(result)
                show_frame_with_border(frame.copy(), result.split()[0])
            formatted_timestamp = format_timestamp(timestamp)
            return f"{key} @ {formatted_timestamp}"
    return None


def format_timestamp(timestamp):
    """
    Format a time float with 2 decimals in seconds to a nicely formatted string in minutes and seconds.

    Args:
        timestamp (float): Time in seconds with 2 decimal places.

    Returns:
        str: Formatted time string (e.g., '1:23' for 1 minute and 23 seconds).
    """
    minutes = int(timestamp // 60)
    seconds = int(timestamp % 60)
    return f"{minutes}:{seconds:02d}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Video template matching")
    parser.add_argument("video_path", nargs="?", help="Path to the input video file")
    parser.add_argument('--show-matches', action='store_true', help='Flag to display frames with matched templates')
    parser.add_argument('--wait', action='store_true', help='Flag to pause terminal after execution')
    parser.add_argument('--extract-interval', type=float, default=DEFAULT_EXTRACT_INTERVAL,
                        help='Time interval (in seconds) between extracted frames')
    parser.add_argument('--time-elapsed', action='store_true', help='Flag to display elapsed time')
    args = parser.parse_args()

    if not args.video_path:
        args.video_path = input("Copy & Paste the path to your video: ")

    # Path to the input video file
    VID_PATH = args.video_path

    # Flag to display frames with matched templates
    SHOW_MATCH = args.show_matches

    # Start the timer if the --time-elapsed flag is set
    start_time = time.time()

    results = extract_frames(VID_PATH, seconds_between_frames=args.extract_interval)

    # Sort the results based on the timestamp
    sorted_results = sorted(results, key=lambda x: int(x.split('@')[1].strip().replace(':', '')))

    # Print the sorted matched templates and their timestamps
    print("\n".join(sorted_results))

    # End the timer if the --time-elapsed flag is set and display the elapsed time
    if args.time_elapsed or True:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed Time: {elapsed_time:.2f} seconds")

    if args.wait:
        input("Press enter to exit...")
