import os
import itk
import dataloader as dl
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from scipy.stats import describe
import SimpleITK as sitk
import utilities as utils


# Define a function for image registration
def register_image(data_loader, fixed_image_path, moving_image_path, fixed_mask_path, moving_mask_path, output_folder, parameter_files, moving_label_path, image_number):
    # Initialize Elastix
    fixed_image = itk.imread(fixed_image_path, itk.F)
    moving_image = itk.imread(moving_image_path, itk.F)
    fixed_mask = itk.imread(fixed_mask_path, itk.UC)
    moving_mask = itk.imread(moving_mask_path, itk.UC)
    moving_label = itk.imread(moving_label_path, itk.US)
    print(moving_mask)
    parameter_object = itk.ParameterObject.New()
    for file in parameter_files:
        parameter_object.AddParameterFile(file)

    # Load Elastix Image Filter Object
    elastix_object = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)
    # elastix_object.SetFixedImage(fixed_image)
    # elastix_object.SetMovingImage(moving_image)
    elastix_object.SetFixedMask(fixed_mask)
    #elastix_object.SetMovingMask(moving_mask)
    elastix_object.SetParameterObject(parameter_object)

    # Set additional options
    elastix_object.SetLogToConsole(False)
    elastix_object.SetOutputDirectory(output_folder)

    # Update filter object (required)
    elastix_object.UpdateLargestPossibleRegion()

    # Results of Registration
    result_image = elastix_object.GetOutput()
    result_transform_parameters = elastix_object.GetTransformParameterObject()
    print(result_transform_parameters)

    #transformix_parameter_object = itk.ParameterObject.New()
    #transformix_parameter_object.AddParameterFile(output_folder+"\\TransformParameters.0.txt")
    #transformix_parameter_object.SetParameter("FinalBSplineInterpolationOrder", "0")
    #transformix_parameter_object.AddParameterFile(output_folder+"\\TransformParameters.1.txt")
    #transformix_parameter_object.SetParameter("FinalBSplineInterpolationOrder", "0")
    #transformix_parameter_object.AddParameterFile(output_folder+"\\TransformParameters.2.txt")
    #transformix_parameter_object.SetParameter("FinalBSplineInterpolationOrder", "0")


    # Load Transformix Object
    result_transform_parameters.SetParameter("FinalBSplineInterpolationOrder", "0")
    transformix_object = itk.TransformixFilter.New(moving_label)
    transformix_object.SetTransformParameterObject(result_transform_parameters)
    transformix_object.SetOutputDirectory(output_folder)

    # Update object (required)
    transformix_object.UpdateLargestPossibleRegion()

    # Results of Transformation
    result_image_transformix = transformix_object.GetOutput()

    # Save the final registered image
    itk.imwrite(result_image, os.path.join(output_folder, "image_registrated.nii"))
    itk.imwrite(result_image_transformix, os.path.join(output_folder, "labelpropagated.nii"))

    # Save the transformation parameter files
    #elastix_object.GetParameterMap()
    #elastix_object.PrintParameterMap(os.path.join(output_folder, "parameter_file.txt"))

def sum_pixel_intensities_all_brain(output_folder, start_number, end_number):
    input_folder = os.getcwd()  
    result_data = None

    for number in range(start_number, end_number + 1):
        folder_name = f"{number}.nii.gz_output"
        folder_path = os.path.join(input_folder, folder_name)
        file_path = os.path.join(folder_path, "image_registrated.nii") #change here the name to image_registrated to get the mean intensities

        if os.path.exists(file_path):
            img = nib.load(file_path)
            data = img.get_fdata()
            if result_data is None:
                result_data = np.zeros_like(data)
            result_data += data

    #uncomment the following line to calculate the mean intensities
    result_data = np.floor(result_data/14)
    if result_data is not None:
        result_img = nib.Nifti1Image(result_data, img.affine)
        result_filename = os.path.join(output_folder, "summed_mean_intensity.nii")
        nib.save(result_img, result_filename)
    else:
        print("No files found for summation.")


def sum_pixel_intensities(output_folder, start_number, end_number):
    input_folder = os.getcwd()  

    result_data = None
    file_count = 0  # Keep track of how many files are processed

    for number in range(start_number, end_number + 1):
        folder_name = f"{number}.nii.gz_output"
        folder_path = os.path.join(input_folder, folder_name)
        image_file_path = os.path.join(folder_path, "image_registrated.nii")
        label_file_path = os.path.join(folder_path, "labelpropagated.nii")  

        if os.path.exists(image_file_path) and os.path.exists(label_file_path):
            img = nib.load(image_file_path)
            label_img = nib.load(label_file_path)

            data = img.get_fdata()
            label_data = label_img.get_fdata()

            # Apply the mask: only keep data where label_data is nonzero
            masked_data = np.where(label_data != 0, data, 0)

            if result_data is None:
                result_data = np.zeros_like(masked_data)
            result_data += masked_data
            file_count += 1

    # Calculate the mean intensities instead of sum if file_count is greater than 0
    if result_data is not None and file_count > 0:
        result_data /= file_count

        # Convert the data type to reduce file size
        result_data = np.array(result_data, dtype=np.float32)  

        result_img = nib.Nifti1Image(result_data, img.affine, img.header)
        result_filename = os.path.join(output_folder, "summed_mean_intensity.nii")
        nib.save(result_img, result_filename)
    else:
        print("No files found for summation.")



def process_and_divide_images(fixed_image, sum_intensities, output_folder):
    fixed_img = nib.load(fixed_image)
    sum_img = nib.load(sum_intensities)
    
    fixed_data = fixed_img.get_fdata()
    sum_data = sum_img.get_fdata()

    # Create masks for values 1, 2, and 3 in the fixed image
    mask1 = (fixed_data == 1)
    mask2 = (fixed_data == 2)
    mask3 = (fixed_data == 3)

    result1 = np.zeros_like(sum_data)
    result2 = np.zeros_like(sum_data)
    result3 = np.zeros_like(sum_data)

    result1[mask1] = np.clip(sum_data[mask1] / 14, 0, 1)
    result2[mask2] = np.clip(sum_data[mask2] / 28, 0, 1)
    result3[mask3] = np.clip(sum_data[mask3] / 42, 0, 1)
    #this is a workariund to force the resulting label mask to be between 0 and 1

    result_img1 = nib.Nifti1Image(result1, fixed_img.affine)
    result_img2 = nib.Nifti1Image(result2, fixed_img.affine)
    result_img3 = nib.Nifti1Image(result3, fixed_img.affine)

    nib.save(result_img1, os.path.join(output_folder, "label1.nii"))
    nib.save(result_img2, os.path.join(output_folder, "label2.nii"))
    nib.save(result_img3, os.path.join(output_folder, "label3.nii"))


def process_label_folders(start_number, end_number, output_folder):
    label_values = [1, 2, 3]  # Labels to sum and divide
    divisors = [14, 14, 14]  # Corresponding division values

    result_data = [None] * len(label_values)

    for number in range(start_number, end_number + 1):
        folder_name = f"{number}.nii.gz_output"
        folder_path = os.path.join(os.getcwd(), folder_name)
        file_path = os.path.join(folder_path, "labelpropagated.nii")

        if os.path.exists(file_path):
            img = nib.load(file_path)
            data = img.get_fdata()

            for i, label_value in enumerate(label_values):
                mask = (data == label_value)
                if result_data[i] is None:
                    result_data[i] = np.zeros_like(data)
                result_data[i][mask] += 1

    for i, label_value in enumerate(label_values):
        if result_data[i] is not None:
            result_data[i] = result_data[i] / divisors[i]

            result_img = nib.Nifti1Image(result_data[i], img.affine)
            result_filename = os.path.join(output_folder, f"smooth{label_value}.nii")
            nib.save(result_img, result_filename)



def plot_tissue_histograms(reference_image_path, reference_label_path):
    # Load the reference image and label data
    reference_image = nib.load(reference_image_path)
    reference_label = nib.load(reference_label_path)

    # Get the image data and label data as NumPy arrays
    image_data = reference_image.get_fdata()
    label_data = reference_label.get_fdata()

    # Initialize labels of interest (1, 2, 3) and corresponding tissue names
    labels = [1, 2, 3]
    tissue_names = ['CSF', 'White Matter', 'Gray Matter']

    fig_individual, axs_individual = plt.subplots(len(labels), 2, figsize=(12, 8))  
    fig_combined, axs_combined = plt.subplots(1, 1, figsize=(8, 6))

    sns.set(style="whitegrid")

    # Create empty lists to store tissue statistics
    tissue_means = []
    tissue_stdevs = []

    normalized_means = []
    normalized_stdevs = []

    # Iterate through the labels of interest
    for i, (label, tissue_name) in enumerate(zip(labels, tissue_names)):
        # Create a binary mask for the current label
        mask = (label_data == label)

        label_intensities = image_data * mask

        x_max = label_intensities.max()
        normalized_intensities = label_intensities / x_max

        # Plot the reference image slice
        slice_index = len(image_data) // 2  # Middle slice
        im = axs_individual[i, 0].imshow(np.rot90(label_intensities[slice_index]), cmap='gray')
        axs_individual[i, 0].set_title(f'{tissue_name} - Slice {slice_index}')
        divider = make_axes_locatable(axs_individual[i, 0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        sns.histplot(label_intensities[label_intensities != 0], bins=100, kde=True, ax=axs_individual[i, 1])
        axs_individual[i, 1].set_title(f'{tissue_name} Histogram with KDE')
        axs_individual[i, 1].set_xlabel("Intensity Values")
        axs_individual[i, 1].set_ylabel("Frequency")

        non_zero_intensities = label_intensities[label_intensities != 0]
        stats = describe(non_zero_intensities)
        mean = stats.mean
        stdev = np.sqrt(stats.variance)
        tissue_means.append(mean)
        tissue_stdevs.append(stdev)
        print(f"{tissue_name} - Non-Normalized Data - Mean: {mean:.2f}, Standard Deviation: {stdev:.2f}")

        # Plot the histogram with KDE line (normalized inten sities data)
        sns.histplot(normalized_intensities[normalized_intensities != 0], bins=100, kde=True, ax=axs_combined)
        axs_combined.set_title("Tissue Distributions with KDE (Normalized Data)")
        axs_combined.set_xlabel("Normalized Intensity Values")
        axs_combined.set_ylabel("Frequency")

        # Calculate and store tissue statistics (normalized intensities data)
        normalized_non_zero_intensities = normalized_intensities[normalized_intensities != 0]
        normalized_stats = describe(normalized_non_zero_intensities)
        normalized_mean = normalized_stats.mean
        normalized_stdev = np.sqrt(normalized_stats.variance)
        normalized_means.append(normalized_mean)
        normalized_stdevs.append(normalized_stdev)
        print(f"{tissue_name} - Normalized Data - Mean: {normalized_mean:.2f}, Standard Deviation: {normalized_stdev:.2f}")
    axs_combined.legend(tissue_names)
    plt.tight_layout()

    plt.show()

    # New code for histograms with both axes normalized and x-values within [0, 1]
    plt.figure(figsize=(12, 8))

    for label, tissue_name in zip(labels, tissue_names):
    # Mask the image data to extract intensities for the label
        label_intensities = image_data * (label_data == label)
        non_zero_intensities = label_intensities[label_intensities != 0]
    
    # Normalize the x-axis values to be between 0 and 1
        normalized_x_values = (non_zero_intensities - non_zero_intensities.min()) / (non_zero_intensities.max() - non_zero_intensities.min())

    # Plot the histogram with both axes normalized
        sns.histplot(non_zero_intensities, bins=100, kde=True, stat="density", label=tissue_name)
        plt.title("Tissue Histograms")
        plt.xlabel("Intensity Values")
        plt.ylabel("Probability Density")

    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_registration_results_with_difference(slice_num, original_path, rigid_path, non_rigid_path, target_path):
    original_img = sitk.ReadImage(original_path, sitk.sitkFloat32)
    rigid_img = sitk.ReadImage(rigid_path, sitk.sitkFloat32)
    non_rigid_img = sitk.ReadImage(non_rigid_path, sitk.sitkFloat32)
    target_img = sitk.ReadImage(target_path, sitk.sitkFloat32)

    # Convert SimpleITK images to numpy arrays
    original_array = sitk.GetArrayFromImage(original_img)
    rigid_array = sitk.GetArrayFromImage(rigid_img)
    non_rigid_array = sitk.GetArrayFromImage(non_rigid_img)
    target_array = sitk.GetArrayFromImage(target_img)
    
    #difference_image = non_rigid_array - original_array

    # Plot the slices
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    axs[0].imshow(original_array[:, slice_num, :], cmap='jet')
    axs[0].set_title('Moving Image')
    axs[0].axis('off')
    
    axs[1].imshow(rigid_array[:, slice_num, :], cmap='jet')
    axs[1].set_title('Rigidly Registered Image')
    axs[1].axis('off')
    
    axs[2].imshow(non_rigid_array[:, slice_num, :], cmap='jet')
    axs[2].set_title('Non-Rigidly Registered Image')
    axs[2].axis('off')
    
    axs[3].imshow(target_array[:, slice_num, :], cmap='jet')
    axs[3].set_title('Fixed Image Image')
    axs[3].axis('off')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()