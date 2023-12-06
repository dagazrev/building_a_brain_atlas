import utilities as utils
import dataloader as dl
import registration as reg
import os


data_folder = "training"
loader = dl.DataLoader(data_folder)

mask_folder = loader.masks_folder


utils.print_image_shapes(mask_folder)
utils.dice_score_explore(mask_folder, 295)

data_folder = "training"  # Replace with the path to your data folder
image_number = 1012  # Replace with the image number
parameter_files = ["Par0033rigid.txt"]  # Replace with your parameter files

data_loader = dl.DataLoader(data_folder)
    
#Get the paths of the fixed image and mask
fixed_image_path, _, _ = data_loader.retrieve_data(image_number)
_,_,fixed_mask_path = data_loader.retrieve_data(image_number)
_,fixed_labels_path,_ = data_loader.retrieve_data(image_number)
    
# Find the paths of all moving images and masks
moving_image_paths = [i for i in data_loader.find_files()[0] if i != f"{image_number}.nii.gz"]
moving_mask_paths = [i for i in data_loader.find_files()[2] if i != f"{image_number}_1C.nii.gz"]
moving_label_paths = [i for i in data_loader.find_files()[1] if i != f"{image_number}_3C.nii.gz"]

# Iterate over moving images and masks for registration
for moving_image, moving_mask, moving_label in zip(moving_image_paths, moving_mask_paths, moving_label_paths):
    moving_image_path = data_loader.images_folder+'\\'+moving_image
    moving_mask_path = data_loader.masks_folder+'\\'+moving_mask
    moving_label_path = data_loader.labels_folder+'\\'+moving_label
    output_folder = f"{moving_image}_output_rig"  # Output folder named after the image number
    os.makedirs(output_folder)
    print(moving_image_path)
    reg.register_image(data_loader, fixed_image_path, moving_image_path, fixed_mask_path, moving_mask_path, output_folder, parameter_files, moving_label_path, image_number)

output_folder = "allprops"  # Replace with the path to your output folder
start_number = 1000  # Replace with the start number
end_number = 1037  # Replace with the end number

reg.sum_pixel_intensities(output_folder, start_number, end_number)

fixed_image = "training\\training-labels\\1012_3C.nii.gz"  # Replace with the path to your fixed image
sum_intensities = "allprops\summed_intensity.nii"  # Replace with the path to your summed intensities


reg.process_and_divide_images(fixed_image, sum_intensities, output_folder)




reg.process_label_folders(start_number, end_number, output_folder)

reference_image_path = 'allprops\summed_mean_intensity_skull_stripped.nii'#"training\\training-images\\1012.nii.gz" this is the og image
reference_label_path = "training\\training-labels\\1012_3C.nii.gz"
reg.plot_tissue_histograms(reference_image_path, reference_label_path)

#reg.plot_registration_results_with_difference(slice_num=100, original_path='training\\training-images\\1006.nii.gz', rigid_path='1006.nii.gz_output_rig\image_registrated.nii', 
#                                          non_rigid_path='1006.nii.gz_output\image_registrated.nii', target_path='training\\training-images\\1012.nii.gz')


