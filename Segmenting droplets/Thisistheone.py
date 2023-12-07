import numpy as np
import tifffile
import cv2
import json
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ImageProcessor:
    def __init__(self, model_path='../Pipeline/model_efficientNetB7Finetune_b_1619.pt'):
        # Load the cell count model
        
        NN = models.efficientnet_b7(weights="IMAGENET1K_V1")

        #resnet not trained later 
        for param in NN.parameters():
            param.requires_grad = False

        num_features = NN.classifier[-1].in_features

        mlp = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 1),
            #nn.Softmax(dim=1)
        )

        NN.classifier = mlp

        NN = NN.to(device)
        
        
        self.segmentation_and_regression_model = NN
        state_dict = torch.load(model_path, map_location = device)
        self.segmentation_and_regression_model.load_state_dict(state_dict)
        self.segmentation_and_regression_model.eval()
       

    def process_image(self, image_path):
        # Load the image from the .tif file
        image = tifffile.imread(image_path)

        # Apply your defined function to separate slices
        droplet_segments, x_coordinates = self.identify_droplet_segments(image)

        # Initialize an empty list to store cell counts and x coordinates
        results = []
        droplet_segment_prelist = []

        # Iterate over slices and pass through the pretrained neural network
        for i, slice in enumerate(droplet_segments):
            slice = slice.astype(np.uint8)
            droplet_segment =  cv2.resize(slice, (128, 128), interpolation=cv2.INTER_LINEAR)
            droplet_segment = np.tile(droplet_segment[np.newaxis, np.newaxis, :, :], (1,3,1,1))
            droplet_segment_tensor = torch.from_numpy(droplet_segment) / 255
            droplet_segment_tensor = droplet_segment_tensor.to(device)

            with torch.no_grad():
                output_cell_count = self.segmentation_and_regression_model(droplet_segment_tensor)

            # Extract ouput
            cell_count = output_cell_count.item()

            # Append results to the list
            results.append({"x_coordinate": x_coordinates[i], "cell_count": cell_count})
            droplet_segment_prelist.append(droplet_segment_tensor)

        return results, droplet_segment_prelist

    def identify_droplet_segments(self, image):

        # 1. Convert the image to greyscale
        grey_image = image[:, :, 0]

        # 2. Apply filtering
        sobel_filtered_image = self.sobel_filter(grey_image)

        # 3. Apply slice on x to get a list of the slices
        segments_x = self.slice_along_x(sobel_filtered_image)

        # setup segments
        image_segments = []
        mean_x_value = []

        for segment_x in segments_x:
            coordinates_y = self.slice_along_y(image = sobel_filtered_image, coordinates= segment_x)
            original_segment = grey_image[coordinates_y[0][0]:coordinates_y[0][1],segment_x[0]:segment_x[1]]
            original_segment = original_segment.astype(int)
            # self.visualize_image(original_segment)
            image_segments.append(original_segment)
            mean_x_value.append(np.mean(segment_x))

        return image_segments, mean_x_value

    def visualize_image(self, image):
        plt.imshow(image, cmap='gray')
        plt.show()

    def sobel_filter(self, gray_arr):
        # Apply Sobel filter to the image
        sobel_x = cv2.Sobel(gray_arr, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_arr, cv2.CV_64F, 0, 1, ksize=3)
        mask = np.sqrt(sobel_x**2 + sobel_y**2)
        return mask

    def slice_along_x(self, image, threshold_start=50, threshold_end=45, min_segment_length=420, max_segment_length=1200):
        # Calculate variance along the x-axis
        x_variances = np.mean(image, axis=0)

        #plt.plot(x_variances)
        #plt.title('Variance along the x-axis')
        #plt.xlabel('X-axis')
        #plt.ylabel('Variance')
        #plt.show()

        # Identify segments where the variance exceeds the threshold
        segments = []
        droplet_started = False
        start_index = 0

        for i, value in enumerate(x_variances):
            if value > threshold_start and not droplet_started:
                droplet_started = True
                start_index = i
            elif value <= threshold_end and droplet_started:
                droplet_started = False
                end_index = i - 1
                segment_length = end_index - start_index

                # Check if the segment length is within the desired range
                if min_segment_length <= segment_length <= max_segment_length:
                    segments.append((start_index, end_index))

        # If a droplet continues to the end of the image, consider it
        if droplet_started:
            end_index = len(x_variances) - 1
            segment_length = end_index - start_index

            # Check if the segment length is within the desired range
            if min_segment_length <= segment_length <= max_segment_length:
                segments.append((start_index, end_index))

        # Extract slices based on the identified positions
        coordinates = [(start, end) for start, end in segments]

        return coordinates

    def slice_along_y(self, image, coordinates, y_threshold_start=40, y_threshold_end=40,
                      min_segment_length=400, max_segment_length=800):
        # Calculate variance along the x-axis
        image = image[:,coordinates[0]:coordinates[1]]
        y_variances = np.mean(image, axis=1)

        #plt.plot(y_variances)
        #plt.title('Variance along the y-axis')
        #plt.xlabel('Y-axis')
        #plt.ylabel('Variance')
        #plt.show()

        # Identify segments where the variance exceeds the threshold
        segments = []
        droplet_started = False
        start_index = 0

        for i, value in enumerate(y_variances):
            if value > y_threshold_start and not droplet_started:
                droplet_started = True
                start_index = i
            elif value <= y_threshold_end and droplet_started:
                droplet_started = False
                end_index = i - 1
                segment_length = end_index - start_index

                # Check if the segment length is within the desired range
                if min_segment_length <= segment_length <= max_segment_length:
                    segments.append((start_index, end_index))

        # Extract slices based on the identified positions
        coordinates_y = [(start, end) for start, end in segments]

        return coordinates_y
    
if __name__ == "__main__":
    inputPath = "../Data/20231206160503Z_test"
    outputPath = "../Data/Output"
    
    # Initialize lists to store results for each frame
    all_results = []
    droplet_segment_tensors_list = []

    # Iterate through each file in inputPath
    for filename in os.listdir(inputPath):
        if filename.endswith(".tif"):
            image_path = os.path.join(inputPath, filename)

            # Process each image
            image_processor = ImageProcessor()
            results, droplet_segment_prelist = image_processor.process_image(image_path)

            separated_tensors_list = []
            for tensor in droplet_segment_prelist:
                # If the tensor has more than one image, split it into individual tensors
                if len(tensor.shape) == 4 and tensor.shape[0] > 1:
                    separated_tensors_list.extend([tensor[i:i+1] for i in range(tensor.shape[0])])
                else:
                    separated_tensors_list.append(tensor)

            droplet_segment_tensors_list.extend(separated_tensors_list)

            # Check if droplet segments are detected
            if results:
                # Append results to Droplet Positions and Cell Counts for the current frame
                frame_results = []
                for result in results:
                    frame_results.append([result["x_coordinate"], result["cell_count"]])

                # Append results for the current frame to the overall list
                all_results.append({f"frame_{filename[:-4]}": frame_results})
            else:
                print(f"No droplet segments detected in {filename}. Skipping.")

    # Create JSON structure with frames sorted numerically
    json_output = {}
    for result in sorted(all_results, key=lambda x: int(list(x.keys())[0].split("_")[1])):
        json_output.update(result)

    save_path = "droplet_segment_tensors.pt"
    torch.save(droplet_segment_tensors_list, save_path)
    print(f"All droplet_segment_tensors saved to {save_path}")

    # Write JSON to file
    output_file_path = os.path.join(outputPath, "output.json")
    with open(output_file_path, 'w') as json_file:
        json.dump(json_output, json_file, indent=2)

    print(f"JSON output written to {output_file_path}")
