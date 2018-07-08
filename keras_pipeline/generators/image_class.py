

from ._image_generator import ImageGenerator


class ImageClassGenerator(ImageGenerator):
    def __init__(self, config):
        # Store general dataset info
        self.dataset         = config.dataset
        self.all_image_index = config.dataset.list_all_image_index()
        self.size            = config.dataset.get_size()
        self.num_classes     = config.dataset.get_num_classes()
        self.label_to_name   = config.dataset.label_to_name

        # Typical generator config
        self.batch_size      = config.batch_size
        self.image_height    = config.image_height
        self.image_width     = config.image_width
        self.stretch_to_fill = config.stretch_to_fill
        self.shuffle         = config.shuffle

        # Create transform generator
        self.transform_parameters = config.transform_parameters
        self.transform_generator = None
        if config.allow_transform:
            self.transform_generator = self._make_transform_generator(config)

        raise NotImplementedError('__init__ is not defined')

        # Validate dataset
        self._validate_dataset()

        # Tools which helps order the data generated
        self.lock = threading.Lock() # this is to allow for parrallel batch processing
        self.group_index_generator = self._make_index_generator()

    def _validate_dataset(self):
        """ Dataset validator which validates the suitability of the dataset """

    def load_X_group(self, group):
        """ Loads the raw inputs from the dataset """
        raise NotImplementedError('load_X_group is not defined')

    def load_Y_group(self, group):
        """ Loads the raw inputs from the dataset """
        raise NotImplementedError('load_Y_group is not defined')

    def preprocess_entry(self, X, Y):
        """ Preprocesses an entry """
        raise NotImplementedError('preprocess_entry is not defined')

    def compute_inputs(self, X_group):
        """ Compute the network inputs """
        raise NotImplementedError('compute_inputs is not defined')

    def compute_targets(self, Y_group):
        """ Compute the network outputs """
        raise NotImplementedError('compute_targets is not defined')
