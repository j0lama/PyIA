from __future__ import division
from PIL import Image
from sklearn import svm
from StringIO import StringIO
import pickle
import sys
import os


def process_directory(directory):
    '''Returns an array of feature vectors for all the image files in a
    directory (and all its subdirectories). Symbolic links are ignored.

    Args:
      directory (str): directory to process.

    Returns:
      list of list of float: a list of feature vectors.
    '''
    training = []
    for root, _, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            img_feature = process_image_file(file_path)
            if img_feature:
                training.append(img_feature)
    return training


def process_image_file(image_path):
    '''Given an image path it returns its feature vector.

    Args:
      image_path (str): path of the image file to process.

    Returns:
      list of float: feature vector on success, None otherwise.
    '''
    image_fp = StringIO(open(image_path, 'rb').read())
    try:
        image = Image.open(image_fp)
        return process_image(image)
    except IOError:
        return None

def process_image(image, blocks=4):
    '''Given a PIL Image object it returns its feature vector.

    Args:
      image (PIL.Image): image to process.
      blocks (int, optional): number of block to subdivide the RGB space into.

    Returns:
      list of float: feature vector if successful. None if the image is not
      RGB.
    '''
    if not image.mode == 'RGB':
        return None
    feature = [0] * blocks * blocks * blocks
    pixel_count = 0
    for pixel in image.getdata():
        ridx = int(pixel[0]/(256/blocks))
        gidx = int(pixel[1]/(256/blocks))
        bidx = int(pixel[2]/(256/blocks))
        idx = ridx + gidx * blocks + bidx * blocks * blocks
        feature[idx] += 1
        pixel_count += 1
    return [x/pixel_count for x in feature]


def train(training_path_a, training_path_b, print_metrics=True):
    '''Trains a classifier. training_path_a and training_path_b should be
    directory paths and each of them should not be a subdirectory of the other
    one. training_path_a and training_path_b are processed by
    process_directory().

    Args:
      training_path_a (str): directory containing sample images of class A.
      training_path_b (str): directory containing sample images of class B.
      print_metrics  (boolean, optional): if True, print statistics about
        classifier performance.

    Returns:
      A classifier (sklearn.svm.SVC).
    '''
    if not os.path.isdir(training_path_a):
        raise IOError('%s is not a directory' % training_path_a)
    if not os.path.isdir(training_path_b):
        raise IOError('%s is not a directory' % training_path_b)
    training_a = process_directory(training_path_a)
    training_b = process_directory(training_path_b)
    # data contains all the training data (a list of feature vectors)
    data = training_a + training_b
    # target is the list of target classes for each feature vector: a '1' for
    # class A and '0' for class B
    target = [0] * len(training_a) + [1] * len(training_b)

    # search for the best classifier within the search space and return it
    clf = svm.SVC(gamma=0.001, C=100.).fit(data, target)
    return clf


def kb_interrupt_handler(classifier):
    with open('classifier', 'wb') as clfFile:
            pickle.dump(classifier, clfFile)
    sys.exit(0)


def main(training_path_a, training_path_b):
    '''Main function. Trains a classifier and allows to use it on images
    downloaded from the Internet.

    Args:
      training_path_a (str): directory containing sample images of class A.
      training_path_b (str): directory containing sample images of class B.
    '''
    classifier = None
    if os.path.isfile('classifier'):
        print('Loading an existing classifier...')
        with open('classifier', 'rb') as clfFile:
            unpickler = pickle.Unpickler(clfFile)
            classifier = unpickler.load()
    else:
        print('Training classifier...')
        classifier = train(training_path_a, training_path_b)
    print('Ready to predict...')
    while True:
        try:
            print("Input an image path: "),
            image_url = raw_input()
            if not image_url:
                break
            features = [process_image_file(image_url)]
            if (classifier.predict(features) == 0):
                print(training_path_a)
            else:
                print(training_path_b)
        except (KeyboardInterrupt, EOFError):
            kb_interrupt_handler(classifier)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: %s [class A images directory] [class B images directory]" %
            sys.argv[0])
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])