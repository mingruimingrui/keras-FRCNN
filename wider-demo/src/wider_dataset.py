import os
from scipy.io import loadmat

from keras_pipeline.datasets._ImageDatasetTemplate import ImageDatasetTemplate
from keras_pipeline.preprocessing.image import read_image

def _load_wider_object_tags():
    return {
        'blur': {
            '0': {'id': 0, 'category': 'clear' , 'tag': 'blur'},
            '1': {'id': 1, 'category': 'normal', 'tag': 'blur'},
            '2': {'id': 2, 'category': 'heavy' , 'tag': 'blur'}
        },
        'expression': {
            '0': {'id': 0, 'category': 'typical'   , 'tag': 'expression'},
            '1': {'id': 1, 'category': 'exaggerate', 'tag': 'expression'}
        },
        'illumination': {
            '0': {'id': 0, 'category': 'normal' , 'tag': 'expression'},
            '1': {'id': 1, 'category': 'extreme', 'tag': 'expression'}
        },
        'occlusion': {
            '0': {'id': 0, 'category': 'none' , 'tag': 'illumination'},
            '1': {'id': 1, 'category': 'partial', 'tag': 'illumination'},
            '2': {'id': 2, 'category': 'heavy' , 'tag': 'illumination'}
        },
        'pose': {
            '0': {'id': 0, 'category': 'typical' , 'tag': 'pose'},
            '1': {'id': 1, 'category': 'atypical', 'tag': 'pose'}
        },
        'invalid': {
            '0': {'id': 0, 'category': 'false', 'tag': 'invalid'},
            '1': {'id': 1, 'category': 'true' , 'tag': 'invalid'}
        },
    }

def _get_object_classes(object_tags):
    object_classes = {}
    count_class = 0
    for tag, tag_info in object_tags.items():
        for class_info in tag.values():
            class_name = class_info['tag'] + '_' + class_info['category']

            class_info['id'] = count_class
            class_info['class_name'] = class_name

            object_classes[class_name] = class_info
            count_class += 1

def _get_object_labels(object_classes):
    object_labels = {}
    for class_name, class_info in object_classes.items():
        object_labels[class_info['id']] = class_info
    return object_labels



class WiderDataset(ImageDatasetTemplate):
    """ Wider dataset API meant to be used by generators.DetectionGenerator
    Requires your wider dataset to be setup in the following format

    root_dir
      |- images
      |   |- 0--Parade
      |   |- ...
      |
      |- annotations

    Args
        root_dir : see above
        set_name : name of the set eg. train

    """

    def __init__(self, root_dir, set_name):
        self.root_dir = root_dir
        self.set_name = set_name

        # Load annotation file
        annotation_file = os.path.join(
            self.root_dir,
            'annotations',
            'wider_face_{}.mat'.format(self.set_name)
        )
        wider = loadmat(annotation_file)

        # Retrieve object class information
        # Here object_labels  is like label to class_name
        #      object_classes is like class_name to label
        # It is also important that label number starts from 0 and ends at num_object_classes
        object_tags = _load_wider_object_tags()
        self.object_classes = _get_object_classes(object_tags)
        self.object_labels  = _get_object_labels (self.object_classes)

        # now get image infos
        # self.image_infos = None
        # raise NotImplementedError('Dataset not implemented')


    """ Required functions no matter which kind of task is at hand """
    def list_image_index(self):
        # Returns list (not numpy array or tuple) of image_index
        raise NotImplementedError('list_image_index method not implemented')

    def get_size(self):
        # Returns number of images in dataset
        raise NotImplementedError('get_size method not implemented')

    def get_num_object_classes(self):
        raise NotImplementedError('get_num_object_classes method not implemented')

    def get_image_aspect_ratio(self, image_index):
        raise NotImplementedError('image_aspect_ratio method not implemented')

    def load_image(self, image_index):
        raise NotImplementedError('load_image method not implemented')

    def load_image_info(self, image_index):
        """ If this is a classification task this needs to have the following attributes
        category_id : corresponding to the image class labels
        """
        raise NotImplementedError('load_image_info method not implemented')

    def load_annotations(self, image_index):
        """ Returns a list of annotations for image with the image_index
        each annotation needs to have the following attributes
            category_id : corresponding to the object class labels
            bbox        : in [x, y, w, h] format
        """
        raise NotImplementedError('load_annotations method not implemented')

    def load_annotations_array(self, image_index):
        """ Returns the annotations in an array format of shape (num_annotes, 5)
        for each annote, the order should be [x, y, w, h, object_class]
        """
        raise NotImplementedError('load_annotations_array method not implemented')

    def object_class_to_label(self, name):
        # label is like an index needs to range from 0 to num_class
        raise NotImplementedError('object_class_to_label method not implemented')

    def label_to_object_class(self, label):
        # label is like an index needs to range from 0 to num_class
        raise NotImplementedError('label_to_object_class method not implemented')
