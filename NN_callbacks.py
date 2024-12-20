from tensorflow.keras.callbacks import Callback

class StopAtMinLR(Callback):
    def __init__(self, monitor='loss', min_lr=1e-6, patience=5, verbose=1):
        super(StopAtMinLR, self).__init__()
        self.monitor = monitor
        self.min_lr = min_lr
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        
        if current is not None:
            # Check if the learning rate has reached the minimum
            if self.model.optimizer.learning_rate <= self.min_lr:
                # If the monitored metric hasn't improved
                if current < self.best:
                    self.best = current
                    self.wait = 0  # Reset wait counter if there's an improvement
                else:
                    self.wait += 1
                    if self.verbose > 0:
                        print(f"Patience {self.wait}/{self.patience}: No improvement in {self.monitor} at min learning rate.")

                    # Stop training if no improvement after reaching min_lr for patience epochs
                    if self.wait >= self.patience:
                        if self.verbose > 0:
                            print(f"Stopping training: {self.monitor} did not improve for {self.patience} epochs at min learning rate.")
                        self.model.stop_training = True
    
class AdjustBatchSizeOnLR(Callback):
    def __init__(self, initial_batch_size, lr_threshold, new_batch_size):
        super().__init__()
        self.initial_batch_size = initial_batch_size
        self.lr_threshold = lr_threshold
        self.new_batch_size = new_batch_size
        self.batch_size_updated = False  # To ensure we only update once

    def on_epoch_end(self, epoch, logs=None):
        current_lr = float(self.model.optimizer.learning_rate.numpy())
        if current_lr <= self.lr_threshold and not self.batch_size_updated:
            print(f"\nLearning rate reached {current_lr}. Increasing batch size to {self.new_batch_size}.")
            self.params['batch_size'] = self.new_batch_size  # Update batch size
            self.batch_size_updated = True

class StopAtLossValue(Callback):
    def __init__(self, target_loss):
        super(StopAtLossValue, self).__init__()
        self.target_loss = target_loss

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('loss')  # Get the training loss
        if current_loss is not None and current_loss <= self.target_loss:
            print(f"\nStopping training: loss has reached {current_loss:.6f}, below the target {self.target_loss:.6f}")
            self.model.stop_training = True