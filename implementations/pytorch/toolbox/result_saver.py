import os

from framework.toolbox.result_saver import ResultSaver

FIELD_DELIMITER = "\t"
HEADER = "ARCHITECTURE" + FIELD_DELIMITER + "LABORATORY" + FIELD_DELIMITER + "EXPERIMENT" + FIELD_DELIMITER + "ERAS" + FIELD_DELIMITER + "EPOCH" + FIELD_DELIMITER + "BATCH" + FIELD_DELIMITER + "MODE" + FIELD_DELIMITER + "MEAUSREMENT"


class PytorchResultSaver(ResultSaver):

    def __init__(self, path):
        super().__init__(path)
        self.__check_file()

    def save_epoch(self, architecture_name: str, experiment_name: str, epoch: int, train_measurement: float,
                   valid_measurement: float):
        self.__write_lines(self.__create_line(architecture_name, experiment_name, epoch, "train", train_measurement))
        self.__write_lines(
            self.__create_line(architecture_name, experiment_name, epoch, "validation", valid_measurement))

    def save_batch(self, architecture_name: str, experiment_name: str, epoch: int, batch: int, measurement: float):
        self.__write_lines(
            self.__create_line(architecture_name, experiment_name, epoch, "train", measurement, str(batch)))

    def __write_lines(self, line):
        with open(self.path, 'a') as file:
            file.write(line + "\n")

    def __create_line(self, architecture_name, experiment_name, epoch, mode, loss, batch=""):
        return architecture_name + FIELD_DELIMITER + self.laboratory_name + FIELD_DELIMITER + experiment_name + FIELD_DELIMITER + str(
            self.eras) + FIELD_DELIMITER + str(
            epoch) + FIELD_DELIMITER + batch + FIELD_DELIMITER + mode + FIELD_DELIMITER + str(loss)

    def __check_file(self):
        if self.__write_header():
            self.__write_lines(HEADER)

    def __write_header(self):
        if not self.__file_exist(): return True
        return self.__is_file_empty()

    def __is_file_empty(self):
        print(os.path.getsize(self.path))
        return os.path.getsize(self.path) == 0

    def __file_exist(self):
        return os.path.exists(self.path)
