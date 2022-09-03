from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay
import imagedatahelpers as idh
import modelsavecallback as msc

# useful constants
EPOCHS = 5
BATCH_SIZE = 32
IMAGE_SIZE = [100, 100]
USE_ALL_DATA = False

if __name__ == "__main__":

    weights_path = './large-files/weights/resnet-fruit-weights'
    train_path = ''
    valid_path = ''
    classes = []

    if USE_ALL_DATA:
        train_path = "./large-files/fruits-360/Training"
        valid_path = "./large-files/fruits-360/Test"

    else:
        # defines the list of classes we're going to work with from the data
        classes = ['Apple Golden 1', 'Avocado', 'Banana', 'Lemon', 'Mango', 'Kiwi', 'Raspberry', 'Strawberry']

        # create the symlinks pointing at training and test data images
        train_path, valid_path = idh.setup_symlinks(classes)

    # create a couple of arrays holding the paths of the training and test images
    image_files = glob(train_path + '/*/*.jp*g')
    valid_image_files = glob(valid_path + '/*/*.jp*g')

    # this will get us a list of category folders
    folders = glob(train_path + '/*')

    if USE_ALL_DATA:
        classes = idh.category_names_from_paths(folders)

    # create the model
    resnet = tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=IMAGE_SIZE + [3],
    )

    # exclude the existing weights from the training
    for layer in resnet.layers:
        layer.trainable = False

    # we always flatten the output of conv layers before passing
    # them into the dense layer
    x = tf.keras.layers.Flatten()(resnet.output)

    # add a dense layer (optional)
    x = tf.keras.layers.Dense(1000, activation='relu')(x)

    # create the final dense layer
    prediction = tf.keras.layers.Dense(len(folders), activation='softmax')(x)

    # create the model
    model = tf.keras.Model(inputs=resnet.input, outputs=prediction)

    # view the model structure
    model.summary()

    # compile the model and provide the loss and optimization method
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics='accuracy')

    # define an image input generator
    gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input
    )

    # create generators for the image data
    # NOTE: we made the loss function sparse_categorical_crossentropy, which means we needed to set the
    #   class_mode parameter to 'sparse' in these generators
    train_generator = gen.flow_from_directory(train_path, target_size=IMAGE_SIZE, shuffle=True, batch_size=BATCH_SIZE, class_mode='sparse')
    valid_generator = gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE, shuffle=True, batch_size=BATCH_SIZE, class_mode='sparse')

    # now do the fit using the generators
    r = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=EPOCHS,
        steps_per_epoch=len(image_files) // BATCH_SIZE,
        validation_steps=len(valid_image_files) // BATCH_SIZE,
        callbacks=[msc.ModelSaveCallback(model, weights_path)]
    )

    # create confusion matrices for the training and validation data
    cm = idh.get_confusion_matrix(
        gen,
        train_path,
        len(image_files),
        batch_size=BATCH_SIZE * 2,
        model=model,
        image_size=IMAGE_SIZE
    )
    print(cm)
    valid_cm = idh.get_confusion_matrix(
        gen,
        valid_path,
        len(valid_image_files),
        batch_size=BATCH_SIZE * 2,
        model=model,
        image_size=IMAGE_SIZE
    )
    print(valid_cm)

    plt.plot(r.history['loss'], label='Train Loss')
    plt.plot(r.history['val_loss'], label='Val. Loss')
    plt.legend()
    plt.show()

    plt.plot(r.history['accuracy'], label='Train Acc.')
    plt.plot(r.history['val_accuracy'], label='Val. Acc.')
    plt.legend()
    plt.show()

    # display the confusion matrices
    # fig, axs = plt.subplots()
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    # TODO: Figure out how to add a title cmd.title('Training Data Confusion Matrix')
    cmd.plot()
    plt.show()

    valid_cmd = ConfusionMatrixDisplay(confusion_matrix=valid_cm, display_labels=classes)
    # TODO: Figure out how to add a title valid_cmd.title('Validation Data Confusion Matrix')
    valid_cmd.plot()
    plt.show()
