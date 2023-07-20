import cv2
import torch
from torchvision import models, transforms


def format_timedelta(td):
    """Utility function to format timedelta objects in a cool way (e.g 00:00:20.05)
    omitting microseconds and retaining milliseconds"""
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return (result + ".00").replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"{result}.{ms:02}".replace(":", "-")


def extract_frames(video_path, frame_interval):
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
            print(f"\r{i}")
            frames.append(frame)

    cap.release()
    return frames


def classify_frames(frames, model):
    results = []
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(frames)
    for frame in frames:
        input_tensor = preprocess(frame)
        input_batch = input_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            input_batch = input_batch.to(device)
            output = model(input_batch)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_probabilities, top_indices = torch.topk(probabilities, k=5)

        results.append((top_indices, top_probabilities))

    return results


def main(path):
    # Extract frames from the video
    frames = extract_frames(path, frame_interval=2)
    print(frames)

    # Classify the frames using the pretrained model
    results = classify_frames(frames, model)

    # Print the top predicted classes and their probabilities for each frame
    for i, result in enumerate(results):
        top_indices, top_probabilities = result
        print(f"Frame {i + 1}:")
        for j in range(len(top_indices)):
            label_index = top_indices[j]
            print(f"Class: {label_index}, Probability: {top_probabilities[j].item()}")


if __name__ == '__main__':
    # Load the pretrained model
    model = models.resnet50(pretrained=True)
    model.eval()  # Set the model to evaluation mode

    # Move the model to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    while True:
        video_path = input("Copy & Paste the path to your video: ")
        main(video_path)
