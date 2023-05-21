import sys
import random
import getopt
import logging
import time
from video_processor import get_videos, process_dataset
import plots
import cv2

import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.applications import EfficientNetB0
from keras.callbacks import EarlyStopping
from keras.models import save_model, load_model

from keras.losses import SparseCategoricalCrossentropy
from keras.layers import Rescaling, TimeDistributed, Dense, GlobalAveragePooling3D, Dropout
import matplotlib.pyplot as plt


def main(argv):
    # Set default values for parameters
    videos = 10
    epochs = 10
    batch_size = 1

    # Get parameters from command line
    try:
        opts, args = getopt.getopt(argv, "v:e:", ["videos=", "epochs="])
    except getopt.GetoptError:
        print("main.py -v <videos> -e <epochs>")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-v", "--videos"):
            videos = int(arg)
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)

    train_index = int(videos * 0.5)
    val_index = int(train_index + (videos * 0.1))
    test_index = int(val_index + (videos * 0.3))
    video_index = int ((train_index + val_index + test_index) // 2)

    # Disable all logging messages
    logging.disable(logging.CRITICAL)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Set GPU memory growth
    def setup_gpu_memory_growth() -> None:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        print(f"{len(physical_devices)} GPU available")

    # Check if GPU is available
    def gpu():
        setup_gpu_memory_growth()
        if tf.test.is_gpu_available():
            print("TensorFlow is running on GPU")
            print("")
        else:
            print("TensorFlow is running on CPU")
            print("")

    if __name__ == '__gpu__':
        gpu()

    print(f"{int(videos)} videos chosen.")
    print(f"{epochs} epochs chosen.")
    print("")
    # time.sleep(1)

    # # Define the healthy and ill paths

    native_paths = ["/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_3_focused",
                      "/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_2_underfocused",
                      "/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_4_overfocused"]

    # "/data/RBC_Phantom_60xOlympus/Donor_1/Native5_focused",
    #                       "/data/RBC_Phantom_60xOlympus/Donor_1/Native5_overfocused2ticks",
    #                       "/data/RBC_Phantom_60xOlympus/Donor_1/Native5_underfocused2ticks"

    modified_paths = ["/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_focused",
                        "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_Overfocused",
                        "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_Underfocused"]

    native_videos, native_labels = get_videos(native_paths, label=1, num_videos=video_index)
    modified_videos, modified_labels = get_videos(modified_paths, label=0, num_videos=video_index)

    # "/data/RBC_Phantom_60xOlympus/Donor_1/FA_0.37wtPercent"
    #

    # v_native_paths = ["/home/raj/PycharmProjects/droplets_video/rbc_classifier/native_mix"]
    # v_modified_paths = ["/home/raj/PycharmProjects/droplets_video/rbc_classifier/mod_mix"]
    #
    # test_native_paths = ["/home/raj/PycharmProjects/droplets_video/rbc_classifier/nat"]
    # test_modified_paths = ["/home/raj/PycharmProjects/droplets_video/rbc_classifier/mod"]

    # # Get the videos and labels.
    # train_native_videos, train_native_labels = get_videos(native_paths, label=1, num_videos=train_index)
    # train_modified_videos, train_modified_labels = get_videos(modified_paths, label=0, num_videos=train_index)
    #
    # val_native_videos, val_native_labels = get_videos(native_paths, label=1, num_videos=val_index)
    # val_modified_videos, val_modified_labels = get_videos(modified_paths, label=0, num_videos=val_index)
    #
    # test_native_videos, test_native_labels = get_videos(native_paths, label=1, num_videos=test_index)
    # test_modified_videos, test_modified_labels = get_videos(modified_paths, label=0, num_videos=test_index)

    # Split the videos and labels into train, validation, and test sets.
    train_native_videos, train_native_labels = native_videos[:train_index], native_labels[:train_index]
    train_modified_videos, train_modified_labels = modified_videos[:train_index], modified_labels[:train_index]

    val_native_videos, val_native_labels = native_videos[train_index:val_index], native_labels[train_index:val_index]
    val_modified_videos, val_modified_labels = modified_videos[train_index:val_index], modified_labels[train_index:val_index]

    test_native_videos, test_native_labels = native_videos[val_index:test_index], native_labels[val_index:test_index]
    test_modified_videos, test_modified_labels = modified_videos[val_index:test_index], modified_labels[val_index:test_index]

    # Split the dataset into train, validation, and test sets.
    train_videos_tensor, train_labels_tensor = process_dataset(train_native_videos, train_modified_videos,
                                                               train_native_labels, train_modified_labels)
    val_videos_tensor, val_labels_tensor = process_dataset(val_native_videos, val_modified_videos,
                                                           val_native_labels, val_modified_labels)
    test_videos_tensor, test_labels_tensor = process_dataset(test_native_videos, test_modified_videos,
                                                             test_native_labels, test_modified_labels)


    # Process the dataset into a form that can be used by the model
    autotune = tf.data.experimental.AUTOTUNE

    train_dataset = tf.data.Dataset.zip((train_videos_tensor, train_labels_tensor))
    train_dataset.cache().shuffle(10).prefetch(buffer_size=autotune)
    train_dataset = train_dataset.batch(batch_size)

    val_dataset = tf.data.Dataset.zip((val_videos_tensor, val_labels_tensor))
    val_dataset.cache().prefetch(buffer_size=autotune)
    val_dataset = val_dataset.batch(batch_size)

    test_dataset = tf.data.Dataset.zip((test_videos_tensor, test_labels_tensor))
    test_dataset.cache().prefetch(buffer_size=autotune)
    test_dataset = test_dataset.batch(batch_size)

    # ----------------------------------- #

    # Load the EfficientNetB0 model without the top layer.
    base_model = EfficientNetB0(include_top=False)
    base_model.trainable = False

    # Create a sequential model.
    model = Sequential([
        # Rescaling the input to the range [0, 1].
        Rescaling(scale=255),
        # TimeDistributed wrapper to apply the same transformation to each sample in the batch.
        TimeDistributed(base_model),
        # Dropout layer to prevent overfitting.
        Dropout(0.2),
        # Dense layer to output the probability of each class.
        Dense(10),
        # Pooling layer to reduce the dimensionality of the input.
        GlobalAveragePooling3D()
    ])

    # Compile the model.
    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Fit the model to the training dataset and validation data.
    history = model.fit(train_dataset, epochs=epochs, batch_size=batch_size, validation_data=val_dataset)

    # # Set the layers with non-serializable objects to None
    # model.layer_with_eager_tensor = None
    #
    # # Save the model
    # model.save('/home/raj/PycharmProjects/model/my_model.h5')

    # Print the final accuracy.
    final_accuracy = history.history['accuracy'][-1] * 100
    print("")
    print("Final training accuracy: {:.2f}%".format(final_accuracy))
    # time.sleep(5)
    print("")

    # Get the training accuracy and validation accuracy from the history object
    training_accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']

    # Get the training loss and validation loss from the history object
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    # Test the model on the test dataset
    test_loss, test_accuracy = model.evaluate(test_dataset)

    # # Load the saved model
    # loaded_model = load_model('/home/raj/PycharmProjects/model/my_model.h5')

    # Save the plot data
    plot_data = {
        'training_accuracy': training_accuracy,
        'validation_accuracy': validation_accuracy,
        'training_loss': training_loss,
        'validation_loss': validation_loss,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy
    }

    np.savez('/home/raj/PycharmProjects/frames/plot_data.npz', **plot_data)

    # # Load the saved plot data
    # loaded_data = np.load('/home/raj/PycharmProjects/frames/plot_data.npz')
    #
    # # Access the individual plot data
    # training_accuracy = loaded_data['training_accuracy']
    # validation_accuracy = loaded_data['validation_accuracy']
    # training_loss = loaded_data['training_loss']
    # validation_loss = loaded_data['validation_loss']
    # test_loss = loaded_data['test_loss']
    # test_accuracy = loaded_data['test_accuracy']

    # Call the plot_accuracy_and_loss function
    plots.plot_accuracy_and_loss(training_accuracy, validation_accuracy, training_loss, validation_loss)

    # # Call the plot_test_accuracy_loss function
    # plots.plot_test_accuracy_loss(test_loss, test_accuracy)

    # Make predictions on the test dataset
    predictions = model.predict(test_dataset)

    # Apply softmax to obtain probabilities
    probabilities = tf.nn.softmax(predictions)

    # Convert probabilities to a NumPy array
    probabilities_array = probabilities.numpy()

    # Normalize probabilities to ensure they total to 100%
    normalized_probabilities = probabilities_array / np.sum(probabilities_array, axis=1, keepdims=True)

    # Extract probabilities of being healthy and ill for each video
    healthy_probabilities = normalized_probabilities[:, 1]
    ill_probabilities = normalized_probabilities[:, 0]

    # Print the probabilities for each video
    for i in range(len(healthy_probabilities)):
        print(f"Video {i + 1}: Probability of being healthy = {healthy_probabilities[i] * 100:.2f}%, Probability of being ill = {ill_probabilities[i] * 100:.2f}%")

    # Find the most accurate healthy and ill probabilities
    max_healthy_prob_index = np.argmax(healthy_probabilities)
    max_ill_prob_index = np.argmax(ill_probabilities)

    # Print the video indices with the most accurate probabilities
    print(f"Video index with highest healthy probability: {max_healthy_prob_index + 1}")
    print(f"Video index with highest ill probability: {max_ill_prob_index + 1}")

    # Call the plot_predictions function
    plots.plot_predictions(predictions, test_videos_tensor)

    # def generate_image_overlay(video_frames):
    #     # Create an empty canvas for the overlay image
    #     overlay_image = np.zeros_like(video_frames[0], dtype=np.float32)
    #
    #     # Iterate through each frame and overlay it on the canvas
    #     for frame in video_frames:
    #         overlay_image = cv2.addWeighted(overlay_image, 1, frame.astype(np.float32), 1, 0)
    #
    #     return overlay_image.astype(np.uint8)
    #
    # # Select a healthy video
    # healthy_video_frames = test_native_videos[0]
    # healthy_overlay_image = generate_image_overlay(healthy_video_frames)
    #
    # # Select a modified video
    # mod_video_frames = test_modified_videos[0]
    # mod_overlay_image = generate_image_overlay(mod_video_frames)
    #
    # # Create a canvas to hold the final plot
    # plot_canvas = np.zeros((healthy_overlay_image.shape[0], healthy_overlay_image.shape[1] + 200, 3), dtype=np.uint8)
    # plot_canvas[:healthy_overlay_image.shape[0], :healthy_overlay_image.shape[1]] = healthy_overlay_image
    #
    # # Draw bars indicating native accuracy and mod accuracy
    # native_accuracy = model.predict(np.expand_dims(test_videos_tensor[0], axis=0))[0][0]
    # mod_accuracy = model.predict(np.expand_dims(test_videos_tensor[video_index], axis=0))[0][1]
    #
    # bar_height = int(plot_canvas.shape[0] / 2)
    # native_bar_width = int(native_accuracy * plot_canvas.shape[1])
    # mod_bar_width = int(mod_accuracy * plot_canvas.shape[1])
    #
    # plot_canvas[:bar_height, healthy_overlay_image.shape[1]:] = (0, 255, 0)  # Native bar (green)
    # plot_canvas[bar_height:, healthy_overlay_image.shape[1]:] = (0, 0, 255)  # Mod bar (red)
    #
    # plot_canvas[:bar_height, healthy_overlay_image.shape[1]:healthy_overlay_image.shape[1] + native_bar_width] = (0, 128, 0)  # Filled part of native bar
    # plot_canvas[bar_height:, healthy_overlay_image.shape[1]:healthy_overlay_image.shape[1] + mod_bar_width] = (0, 0, 128)  # Filled part of mod bar
    #
    # # Display and save the final plot
    # cv2.imwrite('/home/raj/PycharmProjects/frames/overlay_plot.jpg', plot_canvas)

    # Print the test loss and test accuracy.
    print("Test loss: {:.2f}".format(test_loss))
    print("Test accuracy: {:.2f}%".format(test_accuracy * 100))


if __name__ == "__main__":
    main(sys.argv[1:])
