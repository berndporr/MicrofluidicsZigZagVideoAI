import getopt
import sys
import logging
import time
from video_processor import get_videos, process_dataset
import tensorflow as tf

# Set default values for parameters
frame_height = 132
frame_width = 800
num_channels = 3
NUM_FRAMES = 100
TARGET_SHAPE = (132, 800, 3)


def main(argv):
    # Set default values for parameters
    train_videos = 18
    val_videos = 4
    epochs = 10
    batch_size = 1
    # Get parameters from command line
    try:
        opts, args = getopt.getopt(argv, "t:v:e:b:", ["train_videos=", "val_videos=", "epochs=", "batch_size="])
    except getopt.GetoptError:
        print("main.py -t <train_videos> -v <val_videos> -e <epochs> -b <batch_size>")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-t", "--train_videos"):
            train_videos = int(arg)
        elif opt in ("-v", "--val_videos"):
            val_videos = int(arg)
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
        elif opt in ("-b", "--batch_size"):
            batch_size = int(arg)

    # Disable all logging messages
    logging.disable(logging.CRITICAL)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    def setup_gpu_memory_growth() -> None:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        print(f"{len(physical_devices)} GPU available")

    def main():
        setup_gpu_memory_growth()
        if tf.test.is_gpu_available():
            print("TensorFlow is running on GPU")
            print("")
        else:
            print("TensorFlow is running on CPU")
            print("")

    if __name__ == '__main__':
        main()

    print(f"{int(train_videos)} training videos")
    print(f"{int(val_videos)} validation videos")
    print("")
    # time.sleep(1)

    # Define the healthy and ill paths
    healthy_paths = ["/data/RBC_Phantom_60xOlympus/Donor_1/Native5_focused",
                     "/data/RBC_Phantom_60xOlympus/Donor_1/Native5_overfocused2ticks",
                     "/data/RBC_Phantom_60xOlympus/Donor_1/Native5_underfocused2ticks",
                     "/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_3_focused",
                     "/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_2_underfocused",
                     "/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_4_overfocused"]

    ill_paths = ["/data/RBC_Phantom_60xOlympus/Donor_1/FA_0.37wtPercent",
                 "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_focused",
                 "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_Overfocused",
                 "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_Underfocused"]

    # healthy_paths = ["/home/raj/PycharmProjects/droplets_video/rbc_classifier/healthy"]
    # ill_paths = ["/home/raj/PycharmProjects/droplets_video/rbc_classifier/mod2"]

    # Select videos for training and validation sets
    healthy_train_paths, healthy_train_labels = get_videos(healthy_paths, label=1, num_videos=train_videos // 2)
    ill_train_paths, ill_train_labels = get_videos(ill_paths, label=0, num_videos=train_videos // 2)

    healthy_val_paths, healthy_val_labels = get_videos(healthy_paths, label=1, num_videos=val_videos // 2)
    ill_val_paths, ill_val_labels = get_videos(ill_paths, label=0, num_videos=val_videos // 2)

    # Process the data
    t_videos, t_labels = process_dataset(healthy_train_paths, ill_train_paths,
                                            healthy_train_labels, ill_train_labels)

    v_videos, v_labels = process_dataset(healthy_val_paths, ill_val_paths,
                                        healthy_val_labels, ill_val_labels)

    # Convert to tensors
    train_ds = tf.data.Dataset.zip((t_videos, t_labels))
    train_ds = train_ds.shuffle(buffer_size=1000)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    val_ds = tf.data.Dataset.zip((v_videos, v_labels))
    val_ds = val_ds.shuffle(buffer_size=1000)
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)

    # ----------------------------------- #

    # Build the model
    model = tf.keras.Sequential([
        tf.keras.applications.EfficientNetB0(include_top=False),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Print the model summary
    model.summary()
    time.sleep(5)

    # Train the model
    history = model.fit(train_ds, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=val_ds)

    print("")
    accuracy = history.history['accuracy'][-1] * 100
    print('Accuracy: {:.2f}%'.format(accuracy))


if __name__ == "__main__":
    main(sys.argv[1:])
