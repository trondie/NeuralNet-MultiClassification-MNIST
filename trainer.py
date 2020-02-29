import numpy as np
import utils
import matplotlib.pyplot as plt
import typing
from model import cross_entropy_loss, SoftmaxModel, one_hot_encode, pre_process_images
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray,
                       model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    outputs = model.forward(X)
    N = targets.shape[0]
    validations = 0
    for i in range(N):

        # align
        target = np.where(targets[i] == 1)[0][0]
        output = np.argmax(outputs[i])
        if target == output:
            validations += 1
            
    return validations / N

def shuffle(X: np.ndarray, Y: np.ndarray):
    scrambledIndices = np.arange(0, X.shape[0])
    np.random.shuffle(scrambledIndices)
    return X[scrambledIndices], Y[scrambledIndices]

def train(
        model: SoftmaxModel,
        datasets: typing.List[np.ndarray],
        num_epochs: int,
        learning_rate: float,
        batch_size: int,
       
        use_shuffle: bool,
        use_momentum: bool,
        momentum_gamma: float):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = datasets

    # Randomize 
    for i in range(len(model.ws)):
        model.ws[i] = np.random.uniform(-1, 1, size =model.ws[i].shape)
        #model.ws[i] = np.random.normal(0, 1 / np.sqrt(len(model.ws[i])), (len(model.ws[i]), model.neurons_per_layer[i]))

    # Utility variables
    num_batches_per_epoch = X_train.shape[0] // batch_size
    num_steps_per_val = num_batches_per_epoch // 5
    # Tracking variables to track loss / accuracy
    train_loss = {}
    val_loss = {}   
    train_accuracy = {}
    val_accuracy = {}

    global_step = 0
    for epoch in range(num_epochs):
        for step in range(num_batches_per_epoch):
            start = step * batch_size
            end = start + batch_size
            X_batch, Y_batch = X_train[start:end], Y_train[start:end]

            output = model.forward(X_batch)
            model.backward(X_batch, output, Y_batch)

            # Update all
            for i in range(len(model.ws)):
                model.ws[i] -= model.grads[i] * learning_rate

            # Track train / validation loss / accuracy
            # every time we progress 20% through the dataset
            if (global_step % num_steps_per_val) == 0:
                _val_loss = model.forward(X_val)
                val_loss[global_step] = cross_entropy_loss(Y_val, _val_loss)

                _train_loss = model.forward(X_train)
                train_loss[global_step] =  cross_entropy_loss(Y_train, _train_loss)

                train_accuracy[global_step] = calculate_accuracy(
                    X_train, Y_train, model)
                val_accuracy[global_step] = calculate_accuracy(
                    X_val, Y_val, model)

            global_step += 1

        #shuffle
        if use_shuffle == True:
            X_train, Y_train = shuffle(X_train, Y_train)

    return model, train_loss, val_loss, train_accuracy, val_accuracy


if __name__ == "__main__":
    # Load dataset
    validation_percentage = 0.2
    X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_full_mnist(
        validation_percentage)

    # Preprocessing 
    X_train = pre_process_images(X_train)
    X_test = pre_process_images(X_test)
    X_val = pre_process_images(X_val)
    # One hot encoding
    Y_train = one_hot_encode(Y_train, 10)
    Y_test = one_hot_encode(Y_test, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # Hyperparameters
    num_epochs = 20
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [128, 10]
    momentum_gamma = .9  

    use_shuffle = False
    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    model, train_loss, val_loss, train_accuracy, val_accuracy = train(
        model,
        [X_train, Y_train, X_val, Y_val, X_test, Y_test],
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_shuffle=use_shuffle,
        use_momentum=use_momentum,
        momentum_gamma=momentum_gamma)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Test Cross Entropy Loss:",
          cross_entropy_loss(Y_test, model.forward(X_test)))

    print("Final Train accuracy:",
          calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:",
          calculate_accuracy(X_val, Y_val, model))
    print("Final Test accuracy:",
          calculate_accuracy(X_test, Y_test, model))

    # Plot loss
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.ylim([0.0, .5])
    utils.plot_loss(train_loss, "Training Loss")
    utils.plot_loss(val_loss, "Validation Loss")
    plt.xlabel("Number of gradient steps")
    plt.ylabel("Cross Entropy Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    # Plot accuracy
    plt.ylim([0.9, 1.0])
    utils.plot_loss(train_accuracy, "Training Accuracy")
    utils.plot_loss(val_accuracy, "Validation Accuracy")
    plt.legend()
    plt.xlabel("Number of gradient steps")
    plt.ylabel("Accuracy")
    plt.savefig("softmax_train_graph.png")
    plt.show()
