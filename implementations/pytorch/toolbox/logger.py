import os

from framework.toolbox.logger import Logger

FIELD_DELIMITER = "\t"
HEADER = "ARCHITECTURE" + FIELD_DELIMITER + "LABORATORY" + FIELD_DELIMITER + "EXPERIMENT" + FIELD_DELIMITER + "ERAS" + FIELD_DELIMITER + "EPOCH" + FIELD_DELIMITER + "BATCH" + FIELD_DELIMITER + "MODE" + FIELD_DELIMITER + "MEASUREMENT"


class PytorchLogger(Logger):
    def __init__(self, path):
        super().__init__(path)
        self.__check_file()

    def log_epoch(self, architecture: str, experiment: str, epoch: int, train_measurement: float, valid_measurement: float):
        print("epoch")
        self.__write_lines(self.__create_line(architecture, experiment, epoch, "train", train_measurement))
        self.__write_lines(self.__create_line(architecture, experiment, epoch, "validation", valid_measurement))

    def log_batch(self, architecture: str, experiment: str, epoch: int, batch: int, measurement: float):
        self.__write_lines(self.__create_line(architecture, experiment, epoch, "train", measurement, str(batch)))

    def __write_lines(self, line):
        with open(self.path, 'a') as file:
            file.write(line + "\n")

    def __create_line(self, architecture_name, experiment_name, epoch, mode, loss, batch=""):
        return architecture_name + FIELD_DELIMITER + self.laboratory + FIELD_DELIMITER + experiment_name + FIELD_DELIMITER + str(
            self.era) + FIELD_DELIMITER + str(
            epoch) + FIELD_DELIMITER + batch + FIELD_DELIMITER + mode + FIELD_DELIMITER + str(loss)

    def __check_file(self):
        if self.__write_header():
            self.__write_lines(HEADER)

    def __write_header(self):
        if not self.__file_exist():
            return True
        return self.__is_file_empty()

    def __is_file_empty(self):
        print(os.path.getsize(self.path))
        return os.path.getsize(self.path) == 0

    def __file_exist(self):
        return os.path.exists(self.path)
