import getopt
import sys
import logging
import time
from video_processor import get_videos, preprocess_data
import tensorflow as tf
import numpy as np
from keras.utils import to_categorical

TARGET_SHAPE = (132, 800, 3)

def main(argv):
    # Set default values for parameters
    train_videos = 10
    val_videos = 2
    epochs = 5
    batch_size = 1

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
        print(f"{len(physical_devices)} GPU(s) available")

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

    # Select videos for healthy and ill training and validation sets
    healthy_train_paths, healthy_train_labels = get_videos(healthy_paths, label=1, num_videos=train_videos // 2)
    ill_train_paths, ill_train_labels = get_videos(ill_paths, label=0, num_videos=train_videos // 2)

    healthy_val_paths, healthy_val_labels = get_videos(healthy_paths, label=1, num_videos=val_videos // 2)
    ill_val_paths, ill_val_labels = get_videos(ill_paths, label=0, num_videos=val_videos // 2)

    print(f"{len(healthy_train_paths)} healthy training videos sent to the model")
    print(f"{len(ill_train_paths)} ill training videos sent to the model")
    # time.sleep(1)
    print("")

    print(f"{len(healthy_val_paths)} healthy validation videos sent to the model")
    print(f"{len(ill_val_paths)} ill validation videos sent to the model")
    # time.sleep(1)
    print("")

    # Preprocess the training data
    train_data_resized, train_labels = preprocess_data(healthy_train_paths, ill_train_paths,
                                                       healthy_train_labels, ill_train_labels,
                                                       target_shape=TARGET_SHAPE, num_samples=train_videos)

    # Preprocess the validation data
    val_data_resized, val_labels = preprocess_data(healthy_val_paths, ill_val_paths,
                                                   healthy_val_labels, ill_val_labels,
                                                   target_shape=TARGET_SHAPE, num_samples=val_videos)

    train_paths = healthy_train_paths + ill_train_paths
    val_paths = healthy_val_paths + ill_val_paths

    print("")
    time.sleep(1)
    print("train_data_resized shape:", train_data_resized.shape, "dtype:", train_data_resized.dtype)
    print("train_labels shape:", train_labels.shape, "dtype:", train_labels.dtype)
    print("")
    # time.sleep(1)

    # Print 5 random training videos and labels
    print("Random training videos and labels:")
    for _ in range(5):
        idx = np.random.choice(train_videos)
        label = train_labels[idx]
        video_path = train_paths[idx]
        print("Label:", label, "Video path:", video_path)
    print("")
    time.sleep(1)

    print("val_data_resized shape:", val_data_resized.shape, "dtype:", val_data_resized.dtype)
    print("val_labels shape:", val_labels.shape, "dtype:", val_labels.dtype)
    print("")
    # time.sleep(1)

    # Print 5 random validation videos and labels
    print("Random validation videos and labels:")
    for _ in range(5):
        idx = np.random.choice(val_videos)
        label = val_labels[idx]
        video_path = val_paths[idx]
        print("Label:", label, "Video path:", video_path)
    print("")
    time.sleep(1)

    # ----------------------------------- #

    # Define the input shape of the video frames
    num_frames = 100
    frame_height = 132
    frame_width = 800
    num_channels = 3

    inputs = tf.keras.layers.Input(shape=(num_frames, frame_height, frame_width, num_channels))

    # Load the EfficientNetB0 model without the top layer
    net = tf.keras.applications.EfficientNetB0(include_top=False)
    net.trainable = False

    # Add custom layers for classification
    x = tf.keras.layers.TimeDistributed(net)(inputs)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(10, activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    predictions = tf.keras.layers.Dense(2, activation='softmax')(x)

    # Finalize the model
    model = tf.keras.models.Model(inputs=inputs, outputs=predictions)

    # Set the optimizer with a smaller learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    # optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)

    # Compile the model
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    train_labels = to_categorical(train_labels, num_classes=2)
    val_labels = to_categorical(val_labels, num_classes=2)

    train_ds = tf.data.Dataset.from_tensor_slices((train_data_resized, train_labels)).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices((val_data_resized, val_labels)).batch(batch_size)

    # Train the model
    history = model.fit(train_ds, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=val_ds)
    print("")
    accuracy = history.history['accuracy'][0] * 100
    print('Accuracy: {:.2f}%'.format(accuracy))

if __name__ == "__main__":
    main(sys.argv[1:])