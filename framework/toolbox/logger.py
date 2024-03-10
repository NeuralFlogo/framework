import os

FIELD_DELIMITER = "\t"
HEADER = "ARCHITECTURE" + FIELD_DELIMITER + "LABORATORY" + FIELD_DELIMITER + "EXPERIMENT" + FIELD_DELIMITER + "ERAS" + FIELD_DELIMITER + "EPOCH" + FIELD_DELIMITER + "BATCH" + FIELD_DELIMITER + "MODE" + FIELD_DELIMITER + "MEASUREMENT"


class Logger:
    def __init__(self, path):
        self.laboratory = None
        self.era = None
        self.path = path
        self.__check_file()

    def set_laboratory_name(self, name: str):
        self.laboratory = name

    def set_era(self, era: int):
        self.era = era

    def log_epoch(self, architecture: str, experiment: str, epoch: int, train_measurement: float, valid_measurement: float):
        self.__write_lines(self.__create_line(architecture, experiment, epoch, "train", train_measurement))
        self.__write_lines(self.__create_line(architecture, experiment, epoch, "validation", valid_measurement))

    def log_batch(self, architecture: str, experiment: str, epoch: int, batch: int, measurement: float):
        self.__write_lines(self.__create_line(architecture, experiment, epoch, "train", measurement, str(batch)))

    def __write_lines(self, line: str):
        with open(self.path, 'a') as file:
            file.write(line + "\n")

    def __create_line(self, architecture: str, experiment: str, epoch: int, mode: str, loss: float, batch=""):
        return (architecture + FIELD_DELIMITER + self.laboratory + FIELD_DELIMITER + experiment + FIELD_DELIMITER
                + str(self.era) + FIELD_DELIMITER + str(epoch) + FIELD_DELIMITER + batch + FIELD_DELIMITER
                + mode + FIELD_DELIMITER + str(loss))

    def __check_file(self):
        if self.__write_header():
            self.__write_lines(HEADER)

    def __write_header(self):
        if not self.__file_exist():
            return True
        return self.__is_file_empty()

    def __is_file_empty(self):
        return os.path.getsize(self.path) == 0

    def __file_exist(self):
        return os.path.exists(self.path)
