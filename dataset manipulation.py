import os

import cv2
import numpy as np


def create_dataset(image_path, template_path, output_dir, num_samples):
    # Load the template image
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_samples):
        # Load the original image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Generate random translation
        dx, dy = np.random.randint(-20, 20, size=(2,))
        translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))

        # Generate random rotation
        angle = np.random.randint(-10, 10)
        rotation_matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
        rotated_image = cv2.warpAffine(translated_image, rotation_matrix, (image.shape[1], image.shape[0]))

        # Generate random scaling
        scale_factor = np.random.uniform(0.8, 1.2)
        scaled_image = cv2.resize(rotated_image, None, fx=scale_factor, fy=scale_factor)

        # Add the template to the image
        x, y = np.random.randint(0, scaled_image.shape[1] - template.shape[1]), np.random.randint(0, scaled_image.shape[
            0] - template.shape[0])
        scaled_image[y:y + template.shape[0], x:x + template.shape[1]] = template

        # Save the generated image
        output_path = os.path.join(output_dir, f"sample_{i}.png")
        cv2.imwrite(output_path, scaled_image)


if __name__ == '__main__':
    image_path = "C:\\Users\\amazi\\PycharmProjects\\LoLHighlights\\images of phrases\\double kill FULL.png"  # Replace with your original image path
    template_path = "C:\\Users\\amazi\\PycharmProjects\\LoLHighlights\\images of phrases\\double kill.png"  # Replace with your template image path
    output_dir = "C:\\Users\\amazi\\PycharmProjects\\LoLHighlights\\LARGE images of phrases"  # Output directory where generated images will be saved
    num_samples = 100  # Number of samples to generate

    create_dataset(image_path, template_path, output_dir, num_samples)
