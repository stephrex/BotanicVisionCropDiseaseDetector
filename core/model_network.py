import tensorflow as tf
from tensorflow import keras


class CropDiseaseDetectionNet:
    @staticmethod
    def build(num_classes, input_shape=(None, None, 3)):
        input = keras.layers.Input(shape=input_shape)
        resized_input = keras.layers.Resizing(width=224, height=224)(input)
        scaled_input = keras.layers.Rescaling(1/255.)(resized_input)
        x = keras.layers.Conv2D(filters=128, kernel_size=5, strides=2,
                                activation='relu', kernel_initializer='he_normal')(scaled_input)
        x = keras.layers.BatchNormalization()(x)
        inception_module_layer1_conv1 = keras.layers.Conv2D(
            filters=64, kernel_size=1, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        inception_module_layer1_conv2 = keras.layers.Conv2D(
            filters=64, kernel_size=1, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        inception_module_layer1_maxpool = keras.layers.MaxPool2D(
            pool_size=3, strides=1, padding='same')(x)
        inception_module_layer2_conv1 = keras.layers.Conv2D(
            filters=64, kernel_size=1, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        inception_module_layer2_conv2 = keras.layers.Conv2D(
            filters=128, kernel_size=3, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(inception_module_layer1_conv1)
        inception_module_layer2_conv3 = keras.layers.Conv2D(
            filters=256, kernel_size=5, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(inception_module_layer1_conv2)
        inception_module_layer2_conv4 = keras.layers.Conv2D(
            filters=256, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal', activation='relu')(inception_module_layer1_maxpool)
        inception_module_layer3_concatenate = keras.layers.Concatenate(axis=3)(
            [inception_module_layer2_conv1, inception_module_layer2_conv2, inception_module_layer2_conv3, inception_module_layer2_conv4])
        x = keras.layers.BatchNormalization()(inception_module_layer3_concatenate)
        x = keras.layers.Conv2D(filters=256, kernel_size=1, strides=1,
                                activation='relu', kernel_initializer='he_normal')(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        output = keras.layers.Dense(num_classes, activation='softmax')(x)
        model = keras.Model(input, output)
        return model
