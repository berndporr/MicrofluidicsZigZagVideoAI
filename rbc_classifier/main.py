import sys
import getopt
import logging
from video_processor import get_videos, process_dataset

import tensorflow as tf
from keras.models import Sequential
from keras.applications import EfficientNetB0
from keras.callbacks import EarlyStopping
from keras.losses import SparseCategoricalCrossentropy
from keras.layers import Rescaling, TimeDistributed, Dropout, Dense, GlobalAveragePooling3D


def main(argv):
    # Set default values for parameters
    videos = 100
    train_index = int(videos * 0.4)
    val_index = int(videos + (videos * 0.1))
    epochs = 50
    batch_size = 1

    # Get parameters from command line
    try:
        opts, args = getopt.getopt(argv, "v:e:b:", ["videos=", "epochs=", "batch_size="])
    except getopt.GetoptError:
        print("main.py -v <videos> -e <epochs> -b <batch_size>")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-v", "--videos"):
            videos = int(arg)
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
        elif opt in ("-b", "--batch_size"):
            batch_size = int(arg)

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

    print(f"{int(videos)} videos chosen")
    print("")
    # time.sleep(1)

    # Define the healthy and ill paths
    native_paths = ["/home/raj/PycharmProjects/droplets_video/rbc_classifier/native_mix"]
    modified_paths = ["/home/raj/PycharmProjects/droplets_video/rbc_classifier/mod_mix"]

    # native_paths = ["/data/RBC_Phantom_60xOlympus/Donor_1/Native5_focused",
    #                 "/data/RBC_Phantom_60xOlympus/Donor_1/Native5_overfocused2ticks",
    #                 "/data/RBC_Phantom_60xOlympus/Donor_1/Native5_underfocused2ticks",
    #                 "/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_3_focused",
    #                 "/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_2_underfocused",
    #                 "/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_4_overfocused"]
    #
    # modified_paths = ["/data/RBC_Phantom_60xOlympus/Donor_1/FA_0.37wtPercent", 15
    #                   "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_focused",
    #                   "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_Overfocused",
    #                   "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_Underfocused"]

    # Get the videos and labels.
    native_videos, native_labels = get_videos(native_paths, label=1, num_videos=videos // 2)
    modified_videos, modified_labels = get_videos(modified_paths, label=0, num_videos=videos // 2)

    # Split the videos and labels into train, validation, and test sets.
    train_native_videos, train_native_labels = native_videos[:train_index], native_labels[:train_index]
    train_modified_videos, train_modified_labels = modified_videos[:train_index], modified_labels[:train_index]

    val_native_videos, val_native_labels = native_videos[train_index:val_index], native_labels[train_index:val_index]
    val_modified_videos, val_modified_labels = modified_videos[train_index:val_index], modified_labels[train_index:val_index]

    # test_native_videos, test_native_labels = native_videos[val_index:], native_labels[val_index:]
    # test_modified_videos, test_modified_labels = modified_videos[val_index:], modified_labels[val_index:]

    # Split the dataset into train, validation, and test sets.
    train_videos_tensor, train_labels_tensor = process_dataset(train_native_videos, train_modified_videos,
                                                               train_native_labels, train_modified_labels)
    val_videos_tensor, val_labels_tensor = process_dataset(val_native_videos, val_modified_videos,
                                                           val_native_labels, val_modified_labels)
    # test_videos_tensor, test_labels_tensor = process_dataset(test_native_videos, test_modified_videos,
    #                                                          test_native_labels, test_modified_labels)

    # Process the dataset into a form that can be used by the model
    autotune = tf.data.experimental.AUTOTUNE

    train_dataset = tf.data.Dataset.zip((train_videos_tensor, train_labels_tensor))
    train_dataset.cache().shuffle(10).prefetch(buffer_size=autotune)
    train_dataset = train_dataset.batch(batch_size)

    val_dataset = tf.data.Dataset.zip((val_videos_tensor, val_labels_tensor))
    val_dataset.cache().shuffle(10).prefetch(buffer_size=autotune)
    val_dataset = val_dataset.batch(batch_size)

    # test_dataset = tf.data.Dataset.zip((test_videos_tensor, test_labels_tensor))
    # test_dataset.cache().shuffle(10).prefetch(buffer_size=autotune)
    # test_dataset = test_dataset.batch(batch_size)

    # ----------------------------------- #

    # Load the EfficientNetB0 model without the top layer.
    base_model = EfficientNetB0(include_top=False)
    base_model.trainable = False

    # Create a sequential model.
    model = Sequential([
        Rescaling(scale=255),
        TimeDistributed(base_model),
        Dense(10),
        GlobalAveragePooling3D()
    ])

    # Compile the model.
    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Fit the model to the training dataset and validation data.
    model.fit(train_dataset, epochs=epochs, batch_size=batch_size,
              validation_data=val_dataset,
              callbacks=EarlyStopping(patience=5, monitor='val_loss'))

    # # Evaluate the model on the test data
    # test_loss, test_acc = model.evaluate(test_dataset)
    # print('Test accuracy:', test_acc)


if __name__ == "__main__":
    main(sys.argv[1:])
