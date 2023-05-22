import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf

save_directory = '/home/raj/PycharmProjects/frames/'


def plot_accuracy_and_loss(training_accuracy, validation_accuracy, training_loss, validation_loss):
    epochs = range(1, len(training_accuracy) + 1)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the training accuracy on the first subplot
    ax1.plot(epochs, training_accuracy, 'b', label='Training Accuracy')
    ax1.plot(epochs, validation_accuracy, 'r', label='Validation Accuracy')
    ax1.set_title('Training Accuracy vs Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Plot the training loss on the second subplot
    ax2.plot(epochs, training_loss, 'b', label='Training Loss')
    ax2.plot(epochs, validation_loss, 'r', label='Validation Loss')
    ax2.set_title('Training Loss vs Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.3)

    # Save the plot as an EPS file
    plt.savefig(save_directory + 'accuracy_and_loss_plot.eps', format='eps')
    plt.close()


# def plot_test_accuracy_loss(test_loss, test_accuracy):
#     plt.plot(1, test_loss, 'bo', label='Test Loss')
#     plt.plot(1, test_accuracy, 'ro', label='Test Accuracy')
#     plt.title('Test Accuracy and Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Metric')
#     plt.legend()
#     plt.savefig(save_directory + 'test_accuracy_loss_plot.eps', format='eps')
#     plt.close()


def overlay(video, save_directory):
    # Preprocess the video frames
    stacked_frames = tf.stack(list(video), axis=-1)
    average_intensity = tf.reduce_mean(stacked_frames, axis=-1)
    average_intensity_np = average_intensity.numpy()
    output_image = np.uint8(average_intensity_np * 255)
    output_image_gray = cv2.cvtColor(output_image, cv2.COLOR_RGB2GRAY)
    enhanced_image_gray = cv2.equalizeHist(output_image_gray)
    enhanced_image = cv2.cvtColor(enhanced_image_gray, cv2.COLOR_GRAY2RGB)

    # Save the preprocessed image
    cv2.imwrite(save_directory, enhanced_image)

    return enhanced_image


def plot_overlay_and_bar(image, probabilities, save_path):
    # Create the figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    # Plot the overlay image
    axs[0].imshow(image)
    axs[0].set_title('Overlay')
    axs[0].axis('off')

    # Plot the bar chart for the probabilities
    axs[1].bar(['Healthy', 'Ill'], probabilities, color=['green', 'red'])
    axs[1].set_ylim([0, 100])
    axs[1].axis('off')

    # Remove spacing between subplots
    plt.subplots_adjust(wspace=0)

    # Save the figure
    plt.savefig(save_path)


def plot_predictions(predictions, test_videos_tensor):
    # Convert TensorSliceDataset to list
    test_videos_list = list(test_videos_tensor)

    # Find the index of the most accurate healthy and ill probabilities
    max_healthy_index = np.argmax(predictions[:, 1])
    max_ill_index = np.argmax(predictions[:, 0])

    # Extract the videos with the most accurate healthy and ill probabilities
    max_healthy_video = test_videos_list[max_healthy_index]
    max_ill_video = test_videos_list[max_ill_index]

    # Preprocess the videos and save the images
    healthy_img = overlay(max_healthy_video, save_directory + 'max_healthy_video.png')
    ill_img = overlay(max_ill_video, save_directory + 'max_ill_video.png')

    # Get the probabilities of being healthy and ill for the selected videos
    healthy_probabilities = predictions[max_healthy_index].astype(np.float32)
    ill_probabilities = predictions[max_ill_index].astype(np.float32)

    # Normalize probabilities to ensure they total to 100%
    healthy_probabilities = healthy_probabilities / np.sum(healthy_probabilities) * 100
    ill_probabilities = ill_probabilities / np.sum(ill_probabilities) * 100

    # Plot overlay and bar for the most accurate healthy video
    plot_overlay_and_bar(healthy_img, [healthy_probabilities[1], healthy_probabilities[0]], save_directory + 'max_healthy_predictions.png')

    # Plot overlay and bar for the most accurate ill video
    plot_overlay_and_bar(ill_img, [ill_probabilities[1], ill_probabilities[0]], save_directory + 'max_ill_predictions.png')





# def plot_predictions(predictions, test_videos_tensor):
#     # Convert TensorSliceDataset to list
#     test_videos_list = list(test_videos_tensor)
#
#     # Find the index of the most accurate healthy and ill probabilities
#     max_healthy_index = np.argmax(predictions[:, 0])
#     max_ill_index = np.argmax(predictions[:, 1])
#
#     # Extract the videos with the most accurate healthy and ill probabilities
#     max_healthy_video = test_videos_list[max_healthy_index]
#     max_ill_video = test_videos_list[max_ill_index]
#
#     # Preprocess the videos and save the images
#     overlay(max_healthy_video, save_directory + 'max_healthy_video.png')
#     overlay(max_ill_video, save_directory + 'max_ill_video.png')
#
#     # Get the probabilities of being healthy and ill for the selected videos
#     max_healthy_probabilities = predictions[max_healthy_index].astype(np.float32)
#     max_ill_probabilities = predictions[max_ill_index].astype(np.float32)
#
#     # Create the figure and subplots
#     fig, axs = plt.subplots(2, 2, figsize=(12, 10))
#
#     # Load the preprocessed images
#     enhanced_image_max_healthy = cv2.imread(save_directory + 'max_healthy_video.png')
#     enhanced_image_max_ill = cv2.imread(save_directory + 'max_ill_video.png')
#
#     # Plot the healthy video image
#     axs[0, 0].imshow(enhanced_image_max_healthy)
#     axs[0, 0].set_title('Most Accurate Healthy Video')
#     axs[0, 0].axis('off')
#
#     # Plot the bar chart for the probabilities of being healthy or ill for the healthy video
#     axs[0, 1].bar(['Healthy'], max_healthy_probabilities)
#     axs[0, 1].text(0, 0.5, f'Probability: {max_healthy_probabilities[0]:.2%}', verticalalignment='center')
#     axs[0, 1].set_title('Probability Distribution (Healthy Video)')
#     axs[0, 1].set_ylabel('Probability')
#     axs[0, 1].set_ylim([0, 1])
#     axs[0, 1].set_ylim(axs[0, 1].get_ylim()[::-1])  # Reverse the y-axis to match image orientation
#
#     # Scale the bar chart to match the height of the healthy video image
#     image_height = enhanced_image_max_healthy.shape[0]
#     scale_factor = image_height / 132
#     axs[0, 1].set_ylim([0, 1 * scale_factor])
#
#     # Plot the ill video image
#     axs[1, 0].imshow(enhanced_image_max_ill)
#     axs[1, 0].set_title('Most Accurate Ill Video')
#     axs[1, 0].axis('off')
#
#     # Plot the bar chart for the probabilities of being healthy or ill for the ill video
#     axs[1, 1].bar(['Ill'], max_ill_probabilities)
#     axs[1, 1].text(0, 0.5, f'Probability: {max_ill_probabilities[0]:.2%}', verticalalignment='center')
#     axs[1, 1].set_title('Probability Distribution (Ill Video)')
#     axs[1, 1].set_ylabel('Probability')
#     axs[1, 1].set_ylim([0, 1])
#
#     # Adjust the spacing between subplots
#     plt.tight_layout()
#
#     # Save the figure
#     plt.savefig(save_directory + 'predictions.png')

