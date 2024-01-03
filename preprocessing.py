import cv2
import os
from image_processing import func

# Create directories if they do not exist
if not os.path.exists("data2"):
    os.makedirs("data2")
if not os.path.exists("data2/train"):
    os.makedirs("data2/train")
if not os.path.exists("data2/test"):
    os.makedirs("data2/test")

# Define column names for CSV file
column_names = ['label']
for i in range(64 * 64):
    column_names.append("pixel" + str(i))

# Initialize variables
label = 0
var = 0
c1 = 0
c2 = 0

# Iterate through directories and subdirectories in the "train" directory
for (dirpath, dirnames, filenames) in os.walk("train"):
    for dirname in dirnames:
        print(dirname)

        # Create corresponding directories in "data2/train" and "data2/test"
        if not os.path.exists(f"data2/train/{dirname}"):
            os.makedirs(f"data2/train/{dirname}")
        if not os.path.exists(f"data2/test/{dirname}"):
            os.makedirs(f"data2/test/{dirname}")

        # Threshold for splitting images into training and testing sets
        num = 100000000000000000
        i = 0

        for file in os.listdir(f"train/{dirname}"):
            var += 1
            actual_path = f"train/{dirname}/{file}"
            actual_path1 = f"data2/train/{dirname}/{file}"
            actual_path2 = f"data2/test/{dirname}/{file}"

            # Read the image and convert it to black and white using the 'func' function
            img = cv2.imread(actual_path, 0)
            bw_image = func(actual_path)

            # Split images into training and testing sets
            if i < num:
                c1 += 1
                cv2.imwrite(actual_path1, bw_image)
            else:
                c2 += 1
                cv2.imwrite(actual_path2, bw_image)

            i += 1

        label += 1

# Print statistics
print(f"Total number of images processed: {var}")
print(f"Number of images in the training set: {c1}")
print(f"Number of images in the testing set: {c2}")
