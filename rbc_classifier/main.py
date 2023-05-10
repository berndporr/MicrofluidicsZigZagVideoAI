import sys
import getopt
import logging
import time
from video_processor import get_videos, process_dataset

import tensorflow as tf
from keras.models import Sequential
from keras.applications import EfficientNetB0
from keras.callbacks import EarlyStopping
from keras.losses import SparseCategoricalCrossentropy
from keras.layers import Rescaling, TimeDistributed, Dense, GlobalAveragePooling3D, Dropout


def main(argv):
    # Set default values for parameters
    videos = 100
    epochs = 200
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
    val_index = int(videos * 0.1)
    test_index = int(videos * 0.1)

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

    t_native_paths = ["/data/RBC_Phantom_60xOlympus/Donor_1/Native5_focused",
                      "/data/RBC_Phantom_60xOlympus/Donor_1/Native5_overfocused2ticks",
                      "/data/RBC_Phantom_60xOlympus/Donor_1/Native5_underfocused2ticks",
                      "/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_3_focused",
                      "/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_2_underfocused",
                      "/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_4_overfocused"]
    t_modified_paths = ["/data/RBC_Phantom_60xOlympus/Donor_1/FA_0.37wtPercent",
                        "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_focused",
                        "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_Overfocused",
                        "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_Underfocused"]

    v_native_paths = ["/home/raj/PycharmProjects/droplets_video/rbc_classifier/native_mix"]
    v_modified_paths = ["/home/raj/PycharmProjects/droplets_video/rbc_classifier/mod_mix"]

    test_native_paths = ["/home/raj/PycharmProjects/droplets_video/rbc_classifier/nat"]
    test_modified_paths = ["/home/raj/PycharmProjects/droplets_video/rbc_classifier/mod"]

    # Get the videos and labels.
    t_native_videos, t_native_labels = get_videos(t_native_paths, label=1, num_videos=train_index)
    t_modified_videos, t_modified_labels = get_videos(t_modified_paths, label=0, num_videos=train_index)

    v_native_videos, v_native_labels = get_videos(v_native_paths, label=1, num_videos=val_index)
    v_modified_videos, v_modified_labels = get_videos(v_modified_paths, label=0, num_videos=val_index)

    test_native_videos, test_native_labels = get_videos(test_native_paths, label=1, num_videos=test_index)
    test_modified_videos, test_modified_labels = get_videos(test_modified_paths, label=0, num_videos=test_index)

    # Split the videos and labels into train, validation, and test sets.
    train_native_videos, train_native_labels = t_native_videos[:train_index], t_native_labels[:train_index]
    train_modified_videos, train_modified_labels = t_modified_videos[:train_index], t_modified_labels[:train_index]

    val_native_videos, val_native_labels = v_native_videos[:val_index], v_native_labels[:val_index]
    val_modified_videos, val_modified_labels = v_modified_videos[:val_index], v_modified_labels[:val_index]

    test_native_videos, test_native_labels = test_native_videos[:test_index], test_native_labels[:test_index]
    test_modified_videos, test_modified_labels = test_modified_videos[:test_index], test_modified_labels[:test_index]

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

    # Print the final accuracy.
    final_accuracy = history.history['accuracy'][-1] * 100
    print("")
    print("Final training accuracy: {:.2f}%".format(final_accuracy))
    time.sleep(5)
    print("")

    # Test the model on the test dataset.
    test_loss, test_accuracy = model.evaluate(test_dataset)

    # Print the test loss and test accuracy.
    print("Test loss: {:.2f}".format(test_loss))
    print("Test accuracy: {:.2f}%".format(test_accuracy * 100))


if __name__ == "__main__":
    main(sys.argv[1:])
