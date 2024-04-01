import tensorflow as tf
from config import BATCH_SIZE, NUM_EPOCHS, TRAIN_DIR, VAL_DIR, SAVE_MODEL_DIR, RESULTS_VISUALIZATION_DIR
from config import SPECIE_OF_PLANT
from dataloader.datapreprocessing import PlantDiseaseDataset
from core.model_network import CropDiseaseDetectionNet
from evaluate import Evaluate
import wandb

wandb.init(project='Crop_Disease_Detection',
           config={
               'architecture': 'CNN',
               'dataset': 'PlantDieasesDataset',
               'specie_of_plant': SPECIE_OF_PLANT
           })

Train_data = PlantDiseaseDataset(data_dir=TRAIN_DIR).get_images()
Val_data = PlantDiseaseDataset(data_dir=VAL_DIR).get_images()

images, labels = next(Train_data)
num_classes = labels.shape[1]

model = CropDiseaseDetectionNet.build(num_classes=num_classes)
print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

Save_Best_Model_CKPT = tf.keras.callbacks.ModelCheckpoint(
    filepath=SAVE_MODEL_DIR, save_best_only=True, verbose=1)

History = model.fit(Train_data,
                    validation_data=Val_data,
                    batch_size=BATCH_SIZE,
                    epochs=NUM_EPOCHS,
                    callbacks=[Save_Best_Model_CKPT])

for epoch in range(History.epoch):
    wandb.log({"accuracy": History.history['accuracy'][epoch],
               "loss": History.history['loss'][epoch],
               "val_accuracy": History.history['val_accuracy'][epoch],
               "val_loss": History.history['val_loss'][epoch],
               "epoch": epoch})

print('[INFO EVALUATING MODEL')
print(Evaluate.evaluate(VAL_DIR))

print('[INFO] PLOTTING AND SAVING LOSS AND ACCURACY CURVES')
Evaluate.plot_training_loss_curves(output_dir=RESULTS_VISUALIZATION_DIR,
                                   History=History)
