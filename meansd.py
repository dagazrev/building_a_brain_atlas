import numpy as np
import nibabel as nib
import os
import pandas as pd
import matplotlib.pyplot as plt

def calculate_and_output_histograms(testing_folder):
    images_folder = os.path.join(testing_folder, 'training-images')
    labels_folder = os.path.join(testing_folder, 'training-labels')
    tissue_intensities = {'CSF': [], 'WM': [], 'GM': []}
    all_intensities = []

    for image_file in os.listdir(images_folder):
        base_name = image_file.split('.')[0]
        image_path = os.path.join(images_folder, image_file)
        label_path = os.path.join(labels_folder, f"{base_name}_3C.nii.gz")

        image_data = nib.load(image_path).get_fdata()
        label_data = nib.load(label_path).get_fdata()

        for tissue, value in zip(['CSF', 'WM', 'GM'], [1, 2, 3]):
            tissue_mask = label_data == value
            tissue_intensities[tissue].extend(image_data[tissue_mask].flatten())
            all_intensities.extend(image_data[tissue_mask].flatten())
    print(np.max(all_intensities))
    # Find unique intensity values
    unique_intensities = np.unique(all_intensities)

    # Create histograms and calculate mean and standard deviation for each tissue type
    histograms = {}
    for tissue in ['CSF', 'WM', 'GM']:
        hist, _ = np.histogram(tissue_intensities[tissue], bins=unique_intensities, density=True)
        histograms[tissue] = hist

        mean = np.mean(tissue_intensities[tissue])
        std = np.std(tissue_intensities[tissue])
        print(f"{tissue} - Mean: {mean:.2f}, Standard Deviation: {std:.2f}")

    # Normalize histograms
    combined_histogram = np.vstack([histograms[tissue] for tissue in ['CSF', 'WM', 'GM']])
    normalized_histogram = combined_histogram / np.sum(combined_histogram, axis=0)

    # Plotting stacked histograms
    plt.figure(figsize=(10, 6))
    plt.stackplot(unique_intensities[:-1], normalized_histogram, labels=['CSF', 'WM', 'GM'])
    plt.title("Stacked Tissue Type Histograms")
    plt.xlabel("Intensity")
    plt.ylabel("Normalized Probability")
    plt.legend()
    plt.show()

    # Creating DataFrame for CSV output
    df_histograms = pd.DataFrame(normalized_histogram.T, columns=['CSF', 'WM', 'GM'], index=unique_intensities[:-1])
    df_histograms.to_csv('tissue_type_histograms.csv')

    return df_histograms

# Example usage
histogram_df = calculate_and_output_histograms('training')
