import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
import json

save_directory = '/home/raj/PycharmProjects/frames/'


def save_values_to_json(values_dict, json_file_path):
    with open(json_file_path, 'w') as json_file:
        json.dump(values_dict, json_file)


def plot_accuracy_and_loss(training_accuracy, validation_accuracy, training_loss, validation_loss):
    epochs = range(1, len(training_accuracy) + 1)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the training accuracy on the first subplot
    ax1.plot(epochs, training_accuracy, 'b', label='Training acc.')
    ax1.plot(epochs, validation_accuracy, 'r', label='Validation acc.')
    ax1.set_title('Accuracy: Training vs Validation', fontsize=20)
    ax1.set_xlabel('Epochs', fontsize=16)
    ax1.set_ylabel('Accuracy', fontsize=16)
    ax1.legend(fontsize=16)

    # Plot the training loss on the second subplot
    ax2.plot(epochs, training_loss, 'b', label='Training loss')
    ax2.plot(epochs, validation_loss, 'r', label='Validation loss')
    ax2.set_title('Loss:  Training vs Validation', fontsize=20)
    ax2.set_xlabel('Epochs', fontsize=16)
    ax2.set_ylabel('Loss', fontsize=16)
    ax2.legend(fontsize=16)

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.3)

    plt.show()

    # Save the plot as an EPS file
    plt.savefig(save_directory + 'accuracy_and_loss_plot.eps', format='eps')
    plt.close()

    # Save the values as a JSON file
    values_dict = {
        'accuracy_and_loss': {
            'training_accuracy': training_accuracy,
            'validation_accuracy': validation_accuracy,
            'training_loss': training_loss,
            'validation_loss': validation_loss
        }
    }
    save_values_to_json(values_dict, save_directory + 'accuracy_and_loss_values.json')


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

    # Return the enhanced image
    return enhanced_image


def plot_bar_chart(probabilities, save_path):
    # Create a figure with the specified size
    fig, ax = plt.subplots(figsize=(2, 6))

    # Create a bar plot using the exact format of the plot_value_array function
    ax.grid(False)
    ax.set_xticks(range(2))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    thisplot = ax.bar(range(2), [probabilities, 1 - probabilities], color=["green", "red"])
    ax.set_ylim([0, 1])

    # Update the values of the bars
    thisplot[0].set_height(probabilities)
    thisplot[1].set_height(1 - probabilities)

    # Save the plot to the specified path
    plt.savefig(save_path)

    # Close the plot
    plt.close(fig)

    # Save the values as a JSON file
    values_dict = {
        'bar_chart_values': {
            'probabilities': probabilities.tolist(),
            'save_path': save_path
        }
    }
    save_values_to_json(values_dict, save_path[:-4] + '_values.json')


def plot_predictions(predictions, test_videos_tensor):
    # Convert TensorSliceDataset to list
    test_videos_list = list(test_videos_tensor)

    # Apply softmax to obtain probabilities
    probabilities = tf.nn.softmax(predictions)

    # Convert probabilities to a NumPy array
    probabilities_array = probabilities.numpy()

    # Normalize probabilities to ensure they total to 100%
    normalized_probabilities = probabilities_array / np.sum(probabilities_array, axis=1, keepdims=True)

    # Convert the probabilities to float32
    normalized_probabilities = normalized_probabilities.astype(np.float32)

    # Convert the probabilities to a list
    probabilities_list = normalized_probabilities.tolist()

    # Find the indices of the top three healthy and ill probabilities
    top_healthy_indices = np.argsort(probabilities_list, axis=0)[-5:, 1]
    top_ill_indices = np.argsort(probabilities_list, axis=0)[-5:, 0]

    # Extract the videos with the top five healthy and ill probabilities
    top_healthy_videos = [test_videos_list[index] for index in top_healthy_indices]
    top_ill_videos = [test_videos_list[index] for index in top_ill_indices]

    # Preprocess the videos and save the images
    for i, video in enumerate(top_healthy_videos):
        overlay(video, save_directory + f'top_healthy_video_{i + 1}.png')

    for i, video in enumerate(top_ill_videos):
        overlay(video, save_directory + f'top_ill_video_{i + 1}.png')

    # Get the probabilities of being healthy and ill for the selected videos
    top_healthy_probabilities = normalized_probabilities[top_healthy_indices][:, 1]
    top_ill_probabilities = normalized_probabilities[top_ill_indices][:, 0]

    # Create the bar charts for the top five healthy videos
    for i in range(len(top_healthy_videos)):
        plot_bar_chart(top_healthy_probabilities[i], save_directory + f'top_healthy_bar_chart_{i + 1}.eps')

    # Create the bar charts for the top five ill videos
    for i in range(len(top_ill_videos)):
        reversed_probability = 1 - top_ill_probabilities[i]
        plot_bar_chart(reversed_probability, save_directory + f'top_ill_bar_chart_{i + 1}.eps')



