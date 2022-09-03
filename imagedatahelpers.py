import os
from sklearn.metrics import confusion_matrix
import numpy as np


def category_names_from_paths(paths):
    """
    Returns an array containing the names of the last folders in the path list
    which for correctly organised test data should be the category names.
    :param paths:
    :return: An ndarray containing the categories parsed out of the folder names.
    """
    return np.array([p.split(os.path.sep)[-1] for p in paths])


def mkdir(p):
    """
    Creates a new folder at the specified location.
    :param p: Path either absolute or relative to the cwd.
    :return: None
    """
    if not os.path.exists(p):
        os.mkdir(p)


def link_path(src, dst):
    """
    Creates a symlink.
    :param src: Folder to link.
    :param dst: Symlink target folder.
    :return: None
    """
    if not os.path.exists(dst):
        os.symlink(src, dst, target_is_directory=True)


def setup_symlinks(class_names):
    """
    Creates the symlinks needed for the image data.
    :param class_names: A list of classifications that are also folders containing images.
    :return: A tuple containing the training and test data paths.
    """
    mkdir('./large-files/fruits-360-small')
    for sub_folder in ['Training', 'Test']:
        mkdir(f'./large-files/fruits-360-small/{sub_folder}')
        for class_name in class_names:
            src = os.path.realpath(f'./large-files/fruits-360/{sub_folder}/{class_name}', strict=False)
            dst = os.path.realpath(f'./large-files/fruits-360-small/{sub_folder}/{class_name}', strict=False)
            link_path(src, dst)

    # return the training and validation paths
    return os.path.abspath('./large-files/fruits-360-small/Training/'), \
           os.path.abspath('./large-files/fruits-360-small/Test/')


def get_confusion_matrix(generator, data_path, N, image_size, batch_size, model):
    """
    Creates a confusion matrix when generators are being used to create the
    X Y data.
    :param generator: A reference to a ImageDataGenerator that can be used to generate test data.
    :param data_path: A path to the folder containing image data in classification folders.
    :param N: Number of samples in the test data.
    :param image_size: The image size e.g. [100, 100]
    :param batch_size: The batch size to use.
    :param model: A reference to the model to test against.
    :return: A confusion matrix.
    """
    print('Generating confusion matrix', N)

    # will hold the values to pass to sklearn
    predictions = []
    targets = []

    # iterate through the test data
    for x, y in generator.flow_from_directory(
            data_path, target_size=image_size, shuffle=False, batch_size=batch_size):

        # create predictions for this batch
        p = model.predict(x)

        # y and p currently contain N x K arrays with probability values
        # against each classification (K). We need to pull out the index of the
        # class with the highest probability and use that for the input data
        # to the confusion_matrix() function.
        p = np.argmax(p, axis=1)
        y = np.argmax(y, axis=1)

        predictions = np.concatenate((predictions, p))
        targets = np.concatenate((targets, y))

        if len(targets) == N:
            break

    return confusion_matrix(targets, predictions)
