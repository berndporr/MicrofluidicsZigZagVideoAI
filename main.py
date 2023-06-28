#!/usr/bin/python3

import os
import sys
import logging

import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Sequential
from keras.applications import EfficientNetB0
from keras.losses import SparseCategoricalCrossentropy
from keras.layers import Rescaling, TimeDistributed, Dense, GlobalAveragePooling3D, Dropout

import plots
from video_processor import get_videos, save_video_labels_to_file, process_dataset


def logPrint(msg):
    logging.info(msg)
    print(msg)


def main():
    videos = 10
    epochs = 5
    # videos = 200
    # epochs = 100

    if len(sys.argv) < 2:
        print("Usage: {} FA or DA or GA or MIX [-q]".format(sys.argv[0]))
        quit(0)

    option = sys.argv[1]

    # Create the results directory path
    log_directory = os.path.join(os.getcwd(), 'results_' + option)

    # Create the folder if it doesn't exist
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    plots.setResultsDir(log_directory)

    print("Option:", option, "-- results are written into the directory:", log_directory)
    logging.basicConfig(filename=os.path.join(log_directory, "log.txt"),
                        encoding='utf-8',
                        level=logging.INFO,
                        format='%(message)s')

    train_index = int(videos * 0.5)
    val_index = int(train_index + 5)
    test_index = int(val_index + 5)
    # val_index = int(train_index + 50)
    # test_index = int(val_index + 50)
    video_index = int((train_index + val_index + test_index) // 2)

    # Disable logging messages
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

    logPrint("")
    logPrint("{} training videos and {} epochs chosen.".format(int(videos), int(epochs)))
    logPrint("")

    # Define the native and modified paths
    native_paths = [
                       ["/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_3_focused",
                        "/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_2_underfocused",
                        "/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_4_overfocused"]
                   ] * 4

    modified_paths = [
        ["/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay0.74wtPerc_2_IF",
         "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay0.74wtPerc_OF",
         "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_focused",
         "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_Overfocused",
         "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_Underfocused",
         "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_GA1percent_IF",
         "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_GA1percent_OF"],
        ["/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay0.74wtPerc_2_IF",
         "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay0.74wtPerc_OF"],
        ["/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_focused",
         "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_Overfocused",
         "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_Underfocused"],
        ["/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_GA1percent_IF",
         "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_GA1percent_OF"]
    ]

    # Get the selected paths based on the chosen option
    selected_native_paths = native_paths[["MIX", "FA", "DA", "GA"].index(option)]
    logPrint("Native paths: {}".format(selected_native_paths))
    selected_modified_paths = modified_paths[["MIX", "FA", "DA", "GA"].index(option)]
    logPrint("Chem mod paths: {}".format(selected_modified_paths))

    native_videos, native_labels = get_videos(selected_native_paths, label=1, num_videos=video_index)
    modified_videos, modified_labels = get_videos(selected_modified_paths, label=0, num_videos=video_index)

    # Split the videos and labels into train, validation, and test sets.
    train_native_videos, train_native_labels = native_videos[:train_index], native_labels[:train_index]
    train_modified_videos, train_modified_labels = modified_videos[:train_index], modified_labels[:train_index]

    val_native_videos, val_native_labels = native_videos[train_index:val_index], native_labels[train_index:val_index]
    val_modified_videos, val_modified_labels = modified_videos[train_index:val_index], modified_labels[train_index:val_index]

    test_native_videos, test_native_labels = native_videos[val_index:test_index], native_labels[val_index:test_index]
    test_modified_videos, test_modified_labels = modified_videos[val_index:test_index], modified_labels[val_index:test_index]

    # Save videos and labels
    save_video_labels_to_file(os.path.join(log_directory, "train_videos.txt"), train_native_videos + train_modified_videos,
                              train_native_labels + train_modified_labels)
    save_video_labels_to_file(os.path.join(log_directory, "val_videos.txt"), val_native_videos + val_modified_videos,
                              val_native_labels + val_modified_labels)
    save_video_labels_to_file(os.path.join(log_directory, "test_videos.txt"), test_native_videos + test_modified_videos,
                              test_native_labels + test_modified_labels)

    # Split the dataset into train, validation, and test sets.
    train_videos_tensor, train_labels_tensor, train_vid_paths = process_dataset(train_native_videos,
                                                                                train_modified_videos,
                                                                                train_native_labels,
                                                                                train_modified_labels)
    val_videos_tensor, val_labels_tensor, val_vid_paths = process_dataset(val_native_videos,
                                                                          val_modified_videos,
                                                                          val_native_labels,
                                                                          val_modified_labels)
    test_videos_tensor, test_labels_tensor, test_vid_paths = process_dataset(test_native_videos,
                                                                             test_modified_videos,
                                                                             test_native_labels,
                                                                             test_modified_labels)

    # Process the dataset into a form that can be used by the model
    autotune = tf.data.experimental.AUTOTUNE

    train_dataset = tf.data.Dataset.zip((train_videos_tensor, train_labels_tensor))
    train_dataset.cache().shuffle(10).prefetch(buffer_size=autotune)
    train_dataset = train_dataset.batch(1)

    val_dataset = tf.data.Dataset.zip((val_videos_tensor, val_labels_tensor))
    val_dataset.cache().prefetch(buffer_size=autotune)
    val_dataset = val_dataset.batch(1)

    test_dataset = tf.data.Dataset.zip((test_videos_tensor, test_labels_tensor))
    test_dataset.cache().prefetch(buffer_size=autotune)
    test_dataset = test_dataset.batch(1)

    # ----------------------------------- #

    # Load the EfficientNetB0 model without the top layer
    base_model = EfficientNetB0(include_top=False)
    base_model.trainable = False

    # Create a sequential model
    model = Sequential([
        # Rescaling the input to the range [0, 1]
        Rescaling(scale=255),
        # TimeDistributed wrapper to apply the same transformation to each sample in the batch
        TimeDistributed(base_model),
        # Dropout layer to prevent over-fitting
        Dropout(0.2),
        # Dense layer to output the probability of each class
        Dense(10),
        # Pooling layer to reduce the dimensionality of the input
        GlobalAveragePooling3D()
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(log_directory, "model_fit.tsv"), separator="\t")

    # Fit the model to the training dataset and validation data
    history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[csv_logger])

    # Print the final accuracy
    final_accuracy = history.history['accuracy'][-1] * 100
    final_val_accuracy = history.history['val_accuracy'][-1] * 100
    logPrint("")
    logPrint("{} training accuracy: {:.2f}%".format(option, final_accuracy))
    logPrint("{} validation accuracy: {:.2f}%".format(option, final_val_accuracy))
    logPrint("")

    # Get the training accuracy and validation accuracy from the history object
    training_accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']

    # Get the training loss and validation loss from the history object
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    # Test the model on the test dataset
    test_loss, test_accuracy = model.evaluate(test_dataset)

    # Print the test accuracy
    logPrint("")
    logPrint("{} test accuracy: {:.2f}%".format(option, test_accuracy * 100))
    logPrint("")

    # Call the plot_accuracy_and_loss function
    plots.plot_accuracy_and_loss(training_accuracy, validation_accuracy, training_loss, validation_loss)

    # Make predictions on the test dataset
    predictions = model.predict(test_dataset)

    # Call the plot_predictions function
    plots.plot_predictions(predictions, test_videos_tensor, test_vid_paths)

    logging.shutdown()

    print("")
    print("Finished.")
    print("")

    if len(sys.argv) > 2:
        if '-q' in sys.argv[2]:
            return

    plt.show()


if __name__ == "__main__":
    main()
