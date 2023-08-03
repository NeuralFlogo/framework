from discovery.regularization.early_stopping import EarlyStopping
from discovery.regularization.monitors.growth import GrowthMonitor


class TrainingTask:
    def __init__(self, trainer, validator=None, early_stopping: EarlyStopping = EarlyStopping(GrowthMonitor(5, 0.001))):
        self.trainer = trainer
        self.validator = validator
        self.early_stopping = early_stopping

    def execute(self, epochs, architecture, training_dataset, validation_dataset=None):
        for epoch in range(epochs):
            self.trainer.train(architecture, training_dataset)
            if validation_dataset:
                if self.__is_model_trained(architecture, validation_dataset): break
        return architecture

    def __is_model_trained(self, model, validation_dataset):
        return self.early_stopping.check(self.validator.validate(model, validation_dataset))
