from datetime import datetime
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, jaccard_score

def focal_loss(gamma=2., alpha=.75):
    # note alpha > 0.5 weights towards the positive class and vice versa
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon()+pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed

def dice_loss(y_true, y_pred):
    y_true = K.flatten(K.cast(y_true, 'float32')) 
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return 1 - (2 * intersection + K.epsilon()) / (union + K.epsilon())

def iou(y_true, y_pred):
    y_true = K.cast(K.greater(y_true, 0.5), dtype='float32')  
    y_pred = K.cast(K.greater(y_pred, 0.5), dtype='float32') 
    
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    
    return K.mean((intersection + K.epsilon()) / (union + K.epsilon()), axis=0)  # mean IOU over the batch

def plot_history(history, dir_out):
    # plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Loss over epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    plt.savefig(dir_out+'_loss.png')
    plt.show()

    # plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title('Accuracy over epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(loc='upper left')
    plt.savefig(dir_out+'_accuracy.png')
    plt.show()

    # plot IoU
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['iou'], label='Train IoU')
    plt.plot(history.history['val_iou'], label='Validation IoU')
    plt.title('IoU over epochs')
    plt.ylabel('IoU')
    plt.xlabel('Epochs')
    plt.legend(loc='upper left')
    plt.savefig(dir_out+'_iou.png')
    plt.show()


def visualize_prediction(dataset, model):
    def get_random_sample(dataset):
        random_batch = dataset.shuffle(buffer_size=1).take(1)  # shuffle with a buffer size of 1 to get a random batch
        images, masks = next(iter(random_batch))
        index = np.random.randint(0, images.shape[0]) 
        image = images[index].numpy().squeeze()
        mask = masks[index].numpy().squeeze()
        return image, mask
    image, mask = get_random_sample(dataset)
    prediction = model.predict(image[np.newaxis, :, :]).squeeze()
    print(prediction.min(), prediction.mean(), prediction.max())
    prediction_binary = (prediction > 0.5).astype(np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    plot_titles = ['Image', 'Mask', 'Prediction', 'Prediction Binary']

    for ax, img, title in zip(axes, [image, mask, prediction, prediction_binary], plot_titles):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')

def evaluate_model(
    desc_str, 
    test_dataset, 
    model,
    training_time,
    validation_steps,
    steps_per_epoch,
    params,
    threshold=0.5
):
    stop = False
    y_true = []
    y_pred = []
    max_iterations=validation_steps*steps_per_epoch
    num_iterations = 0
    for image_batch, label_batch in test_dataset:
        predictions = model.predict(image_batch)

        for i in range(predictions.shape[0]):
            ground_truth = np.squeeze(label_batch[i].numpy()).flatten()
            prediction = np.squeeze(predictions[i]).flatten()

            y_true.append(ground_truth)
            y_pred.append(prediction > threshold)
            num_iterations += 1
            if max_iterations is not None and num_iterations >= max_iterations:
                stop = True  
                break  
        if stop:
            break

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    iou = jaccard_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    model_info = {'description': [desc_str],
                  'training_time': [training_time],
                  'params': [params],
                  'accuracy': [accuracy],
                  'iou': [iou],
                  'f1_score': [f1],
                  'precision': [precision],
                  'recall': [recall]}

    return model_info