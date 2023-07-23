import cv2
import numpy as np

VID_PATH = None
EXTRACT_INTERVAL = 2
MATCH_THRESH = 0.8
SHOW_MATCH = True

# Template images in grayscale
template_images = {
    "double": cv2.imread('images of phrases\double 2.png', cv2.IMREAD_GRAYSCALE),
    "triple": cv2.imread('images of phrases\\triple 2.png', cv2.IMREAD_GRAYSCALE),
    "quadra": cv2.imread('images of phrases\quadra 2.png', cv2.IMREAD_GRAYSCALE),
    "penta": cv2.imread('images of phrases\penta 2.png', cv2.IMREAD_GRAYSCALE),
}


def extract_frames(video_path, frame_interval):
    """
    Extract frames from the video at the given frame interval.

    Args:
        video_path (str): Path to the video file.
        frame_interval (int): Interval in seconds between extracted frames.

    Returns:
        list: List of extracted frames in grayscale.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval_frames = int(fps * frame_interval)

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        if i % interval_frames == 0:
            # Convert the frame to grayscale
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame_gray)

    cap.release()
    return frames


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


def classify_frames(frames):
    """
    Classify frames using template matching.

    Args:
        frames (list): List of frames in grayscale.

    Returns:
        list: List of strings indicating the matched templates and their timestamps.
    """
    results = []
    for i, frame in enumerate(frames):
        for key in template_images:
            result = cv2.matchTemplate(frame, template_images[key], cv2.TM_CCOEFF_NORMED)
            loc = np.where(result >= MATCH_THRESH)

            # Output True if there is a match, False otherwise
            if len(loc[0]) > 0:
                results.append(f"{key} found at {i * EXTRACT_INTERVAL} seconds")
                if SHOW_MATCH:
                    show_frame_with_border(frame.copy(), key)
                break
    return results


if __name__ == '__main__':
    if not VID_PATH:
        VID_PATH = input("Copy & Paste the path to your video: ")

    # Extract frames from the video
    frames = extract_frames(VID_PATH, frame_interval=EXTRACT_INTERVAL)

    # Classify the frames using template matching
    results = classify_frames(frames)

    # Print the matched templates and their timestamps
    for result in results:
        print(result)
