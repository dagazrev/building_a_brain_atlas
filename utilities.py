import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

def dice_coefficient_test(mask1_path, mask2_path):
    # Read the NIfTI files using SimpleITK
    mask1_itk = sitk.ReadImage(mask1_path)
    mask2_itk = sitk.ReadImage(mask2_path)
    
    # Convert SimpleITK images to numpy arrays
    mask1_array = sitk.GetArrayFromImage(mask1_itk)
    mask2_array = sitk.GetArrayFromImage(mask2_itk)

    # Calculate intersection and total
    intersection = (mask1_array & mask2_array).sum()
    total = mask1_array.sum() + mask2_array.sum()

    # Calculate Dice coefficient
    if total == 0:
        return 1.0  # Dice coefficient is 1 if both masks are empty
    else:
        return 2.0 * intersection / total

def calculate_dice_scores(mask_folder):
    mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.nii.gz')])

    num_masks = len(mask_files)

    # Create a table to store the results
    dice_table = [["Image Index", "Dice Score"]]

    for i in range(num_masks):
        dice_scores = []

        for j in range(num_masks):
            if i != j:  # Exclude the image from itself
                dice = dice_coefficient_test(os.path.join(mask_folder, mask_files[i]), os.path.join(mask_folder, mask_files[j]))
                dice_scores.append(dice)

        # Calculate the average Dice score for the image against the rest
        average_dice = sum(dice_scores) / len(dice_scores)

        dice_table.append([mask_files[i], f"{average_dice:.4f}"])

    # Print the results as a table
    for row in dice_table:
        print("{:<15} {:<10}".format(*row))


def print_image_shapes(folder_name):
    # Get a list of all files in the folder
    image_files = [f for f in os.listdir(folder_name) if f.endswith('.nii.gz')]

    if not image_files:
        print("No image files found in the folder.")
        return

    y_sizes = []

    for image_file in image_files:
        image_path = os.path.join(folder_name, image_file)

        # Read the image and get its shape
        image = sitk.ReadImage(image_path)
        image_size = image.GetSize()
        y_size = image_size[1]  # Y-axis size

        y_sizes.append(y_size)

        print(f"File: {image_file}, Y-axis Size: {y_size}")

    # Calculate statistical summary
    mean_y = np.mean(y_sizes)
    median_y = np.median(y_sizes)
    std_y = np.std(y_sizes)

    print("\nStatistical Summary:")
    print(f"Mean Y-axis Size: {mean_y:.2f}")
    print(f"Median Y-axis Size: {median_y:.2f}")
    print(f"Standard Deviation of Y-axis Size: {std_y:.2f}")

    # Plot a histogram of Y-axis sizes
    plt.hist(y_sizes, bins=20, color='lightblue', edgecolor='black')
    plt.title("Distribution of Y-axis Sizes")
    plt.xlabel("Y-axis Size")
    plt.ylabel("Frequency")
    plt.show()

def resize_image(image, new_size):
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    return resampler.Execute(image)

def dice_score_explore(folder, new_y_size):
    mask_files = sorted([f for f in os.listdir(folder) if f.endswith('.nii.gz')])

    num_masks = len(mask_files)

    # Create a table to store the results (including mean and median columns)
    dice_table = [["Image Index", "Mean Dice Score", "Median Dice Score"]]

    # Create lists to store resized images and Dice scores
    resized_images = []
    dice_scores = []

    for mask_file in mask_files:
        image_path = os.path.join(folder, mask_file)

        # Read the image
        image = sitk.ReadImage(image_path)

        # Resize the image to the specified new Y size
        original_size = list(image.GetSize())
        original_size[1] = new_y_size  # Set the new Y size
        resized_image = resize_image(image, original_size)
        resized_images.append(resized_image)

    for i, mask_file in enumerate(mask_files):
        # Calculate Dice scores for each image against the rest
        dice_scores = []

        for j, other_mask_file in enumerate(mask_files):
            if j != i:  # Exclude the image from itself
                dice = dice_coefficient_test(resized_images[i], resized_images[j])
                dice_scores.append(dice)

        # Calculate the average Dice score for the image against the rest
        average_dice = sum(dice_scores) / len(dice_scores)

        # Calculate the median Dice score
        median_dice = np.median(dice_scores)

        dice_table.append([mask_file, f"{average_dice:.4f}", f"{median_dice:.4f}"])

    # Print the results as a table
    for row in dice_table:
        print("{:<15} {:<15} {:<15}".format(*row))