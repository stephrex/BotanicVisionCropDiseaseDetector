import tensorflow as tf


class PlantDiseaseDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def image_generator(self, data_dir):
        '''
        This method calls the ImageDateGeneratpr dunction of keras, and then perfomr data augmentation on it,
        before preprocessing
        '''
        generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=0.5,
                                                                    width_shift_range=0.3,
                                                                    height_shift_range=0.3,
                                                                    brightness_range=None,
                                                                    shear_range=0.5,
                                                                    zoom_range=0.3,
                                                                    channel_shift_range=0.3,
                                                                    horizontal_flip=True,
                                                                    vertical_flip=True)
        return generator.flow_from_directory(directory=data_dir,
                                             batch_size=32,
                                             shuffle=True,
                                             class_mode='categorical')

    def get_images(self):
        '''
        This Fucntion call the image genrator method, and returns batched of the generator
        '''
        generator = self.image_generator(self.data_dir)
        for batch in generator:
            yield batch
