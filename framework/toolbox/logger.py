import os

Delimiter = "\t"
Header = "ARCHITECTURE" + Delimiter + "LABORATORY" + Delimiter + "EXPERIMENT" + Delimiter + "ERAS" + Delimiter + "EPOCH" + Delimiter + "BATCH" + Delimiter + "MODE" + Delimiter + "MEASUREMENT"


class Logger:
    def __init__(self, path):
        self.laboratory = None
        self.era = None
        self.path = path
        self.__check_file()

    def set_laboratory_name(self, name: str): #TODO pass it as argument not declared
        self.laboratory = name

    def set_era(self, era: int): #TODO pass it as argument not declared
        self.era = era

    def log_epoch(self, architecture: str, experiment: str, epoch: int, train_measurement: float, valid_measurement: float):
        self.__write_lines(self.__entry_of(architecture, experiment, "train", str(train_measurement), str(epoch)))
        self.__write_lines(self.__entry_of(architecture, experiment, "validation", str(valid_measurement), str(epoch)))

    def log_batch(self, architecture: str, experiment: str, epoch: int, batch: int, measurement: float):
        self.__write_lines(self.__entry_of(architecture, experiment, "train", str(measurement), str(epoch), str(batch)))

    def log_test(self, architecture: str, experiment: str, type_measurement: str, measurement: float):
        self.__write_lines(self.__entry_of(architecture, experiment, "test-" + type_measurement, str(measurement)))

    def __write_lines(self, line: str):
        with open(self.path, 'a') as file:
            file.write(line + "\n")

    def __entry_of(self, architecture: str, experiment: str, mode: str, value: str, epoch: str = "", batch: str = ""):
        return Delimiter.join([architecture, self.laboratory, experiment, str(self.era), epoch, batch, mode, value])

    def __check_file(self):
        if self.__write_header():
            self.__write_lines(Header)

    def __write_header(self):
        if not self.__file_exist():
            return True
        return self.__is_file_empty()

    def __is_file_empty(self):
        return os.path.getsize(self.path) == 0

    def __file_exist(self):
        return os.path.exists(self.path)
