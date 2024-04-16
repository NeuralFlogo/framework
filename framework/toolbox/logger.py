import os

Delimiter = "\t"
Header = Delimiter.join(["ARCHITECTURE", "LABORATORY", "EXPERIMENT", "ERAS", "EPOCH", "BATCH", "MODE", "MEASUREMENT"])


class Logger:
    def __init__(self, path):
        self.path = path
        self.laboratory = None
        self.era = None
        self.__init_file()

    def set_laboratory_name(self, name: str):
        self.laboratory = name

    def set_era(self, era: int):
        if era == -1: era = ""
        self.era = str(era)

    def log_epoch(self, architecture: str, experiment: str, epoch: int, train_measurement: float, valid_measurement: float):
        self.__log(self.__entry_of(architecture, experiment, str(epoch), "", "train", str(train_measurement)))
        self.__log(self.__entry_of(architecture, experiment, str(epoch), "", "validation", str(valid_measurement)))

    def log_batch(self, architecture: str, experiment: str, epoch: int, batch: int, measurement: float):
        self.__log(self.__entry_of(architecture, experiment, str(epoch), str(batch), "train", str(measurement)))

    def log_test(self, architecture: str, experiment: str, strategy: str, measurement: float):
        self.__log(self.__entry_of(architecture, experiment, "", "", "{0}-test".format(strategy), str(measurement)))

    def __init_file(self):
        if not self.__file_exist() or self.__is_file_empty():
            self.__log(Header)

    def __log(self, line: str):
        with open(self.path, 'a') as file:
            file.write(line + "\n")

    def __file_exist(self):
        return os.path.exists(self.path)

    def __is_file_empty(self):
        return os.path.getsize(self.path) == 0

    def __entry_of(self, architecture: str, experiment: str, epoch: str, batch: str, mode: str, value: str):
        return Delimiter.join([architecture, self.laboratory, experiment, self.era, epoch, batch, mode, value])
