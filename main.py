import argparse
import os

import cv2
import numpy as np

# Set the desired resolution (1080p)
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# Time interval (in seconds) between extracted frames
DEFAULT_EXTRACT_INTERVAL = 2

# Threshold value for template matching
MATCH_THRESH = 0.85

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Template images in grayscale
template_images = {
    "double": cv2.imread(os.path.join(current_dir, 'data\\images of phrases', 'double.png'), cv2.IMREAD_GRAYSCALE),
    "triple": cv2.imread(os.path.join(current_dir, 'data\\images of phrases', 'triple.png'), cv2.IMREAD_GRAYSCALE),
    "quadra": cv2.imread(os.path.join(current_dir, 'data\\images of phrases', 'quadra.png'), cv2.IMREAD_GRAYSCALE),
    "penta": cv2.imread(os.path.join(current_dir, 'data\\images of phrases', 'penta.png'), cv2.IMREAD_GRAYSCALE),
}


def classify_frames(video_path, frame_interval=DEFAULT_EXTRACT_INTERVAL):
    """
    Classify frames from the video at the given frame interval.

    Args:
        video_path (str): Path to the video file.
        frame_interval (int): Interval in seconds between extracted frames.

    Returns:
        list: List of extracted frames in grayscale.
    """
    results = []
    cap = cv2.VideoCapture(video_path)

    # Set the desired resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval_frames = int(fps * frame_interval)

    for i in range(0, frame_count, interval_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        classification = classify_frame(frame_gray, i // interval_frames * frame_interval)
        if classification:
            results.append(classification)

    cap.release()
    return results


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
    args = parser.parse_args()

    if not args.video_path:
        args.video_path = input("Copy & Paste the path to your video: ")

    # Path to the input video file
    VID_PATH = args.video_path

    # Flag to display frames with matched templates
    SHOW_MATCH = args.show_matches

    results = classify_frames(VID_PATH, frame_interval=args.extract_interval)

    # Print the matched templates and their timestamps
    print("\n".join(results))

    if args.wait:
        input("Press enter to exit...")
