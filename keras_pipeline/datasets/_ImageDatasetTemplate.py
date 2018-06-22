""" Template used for image datasets (no functions actually implemented)
This is merely a guide to what an ImageDataset class should look like

Should be an API interface to load images and other corresponding information from a locally existing dataset
Think of pycocotools but capable of loading images as well

Used as input to generators.generator so we won't have to edit that ever

NOTICE: At this point of writing, all the tools made are made specifically with the intentions of
object detection/segmention and classification in mind
"""


class ImageDatasetTemplate:
    def __init__(self):
        """ Recommended to have the following attributes """
        self.image_infos = None

        """ If image classification task """
        self.image_classes = None

        """ If object detection/segmentation task """
        self.object_classes = None

        raise NotImplementedError('Dataset not implemented')


    """ Required functions no matter which kind of task is at hand """
    def list_image_index(self):
        # Returns list (not numpy array or tuple) of image_index
        raise NotImplementedError('list_image_index method not implemented')

    def get_size(self):
        # Returns number of images in dataset
        raise NotImplementedError('get_size method not implemented')

    def get_image_aspect_ratio(self, image_index):
        raise NotImplementedError('image_aspect_ratio method not implemented')

    def load_image(self, image_index):
        raise NotImplementedError('load_image method not implemented')

    def load_image_info(self, image_index):
        """ If this is a classification task this needs to have the following attributes
        category_ids : corresponding to the image class labels
                       (though this is mainly to help implement load_image_class_array)
        """
        raise NotImplementedError('load_image_info method not implemented')


    """ Required Image Classification functions """
    def get_num_image_classes(self):
        raise NotImplementedError('get_num_image_classes method not implemented')

    def load_image_class_array(self, image_index):
        """ Returns the iamge classes in an array form
        with be a (-1,) shaped array with the length being the number of classes
        associated to the image_index
        """
        raise NotImplementedError('load_image_class_array method not implemented')

    def image_class_to_label(self, name):
        # label is like an index needs to range from 0 to num_class
        raise NotImplementedError('image_class_to_label method not implemented')

    def label_to_image_class(self, label):
        # label is like an index needs to range from 0 to num_class
        raise NotImplementedError('label_to_image_class method not implemented')


    """ Reqruired Object detection/segmentation specific functions """
    def get_num_object_classes(self):
        raise NotImplementedError('get_num_object_classes method not implemented')

    def load_annotations(self, image_index):
        """ Returns a list of annotations for image with the image_index
        each annotation needs to have the following attributes
            category_id : corresponding to the object class labels
                          (though this is mainly to help implement load_annotations_array)
            bbox        : in [x, y, w, h] format
        """
        raise NotImplementedError('load_annotations method not implemented')

    def load_annotations_array(self, image_index):
        """ Returns the annotations in an array format of shape (num_annotes, 5)
        for each annote, the order should be [x, y, w, h, object_class]
        """
        raise NotImplementedError('load_annotations_array method not implemented')

    def object_class_to_object_class_id(self, name):
        raise NotImplementedError('object_class_to_object_class_id method not implemented')

    def object_class_id_to_object_class(self, id):
        raise NotImplementedError('object_class_id_to_object_class method not implemented')
