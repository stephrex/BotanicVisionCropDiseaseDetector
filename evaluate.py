import matplotlib.pyplot as plt
from config import VAL_DIR


class Evaluate:
    @staticmethod
    def evaluate(val_dir):
        model_eval = model.evaluate(VAL_DIR)
        return model_eval

    def plot_training_loss_curves(output_dir, History):
        History = History
        plt.style.use('ggplot')
        plt.figure()
        plt.plot(History.epoch, History.history['loss'], label='train_loss')
        plt.plot(History.epoch,
                 History.history['accuracy'], label='train_accuracy')
        plt.plot(History.epoch,
                 History.history['val_accuracy'], label='val_accuracy')
        plt.plot(History.epoch, History.history['val_loss'], label='val_loss')
        plt.xlabel('epoch')
        plt.ylabel('Loss/Accuracy')
        plt.legend()
        plt.title('Training/Loss History Curves')
        plt.savefig(output_dir, format='png')
