import time
import keras.backend as K
import numpy as np


def cyclic_rate():

    max_lr = settings.learning_rate
    base_lr = settings.learning_rate / 100
    K.set_value(model.optimizer.lr, base_lr)
    step_size = settings.n_batch_per_epoch
    list_lr = [base_lr]
    it = 0
    gamma = 0.99995
    for e in range(settings.nb_epoch):
        # Initialize batch counter
        batch_counter = 1
        start = time.time()

        list_train_loss = []

        while batch_counter < settings.n_batch_per_epoch:

            X_batch, y_batch = queue.get()

            # Train
            train_loss = model.train_on_batch(X_batch, y_batch)
            list_train_loss.append(train_loss)

            batch_counter += 1

            it += 1

            # Learning rate scheduler
            lr = K.get_value(model.optimizer.lr)

            cycle = np.floor(1 + it / (2. * step_size))
            x = np.abs(it / float(step_size) - 2. * cycle + 1)
            lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) * gamma**(it)  # / float(2**(cycle - 1))
            list_lr.append(lr)
            K.set_value(model.optimizer.lr, lr)
