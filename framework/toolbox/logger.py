import os

Delimiter = "\t"
Header = "ARCHITECTURE" + Delimiter + "LABORATORY" + Delimiter + "EXPERIMENT" + Delimiter + "ERAS" + Delimiter + "EPOCH" + Delimiter + "BATCH" + Delimiter + "MODE" + Delimiter + "MEASUREMENT"


class Logger:
    def __init__(self, path):
        self.path = path
        self.__check_file()

    def log_epoch(self, architecture: str, laboratory_name: str, experiment: str, era: int, epoch: int, train_measurement: float, valid_measurement: float):
        self.__write_lines(self.__entry_of(architecture, laboratory_name, experiment, "train", str(train_measurement), str(era), str(epoch)))
        self.__write_lines(self.__entry_of(architecture, laboratory_name, experiment, "validation", str(valid_measurement), str(era), str(epoch)))

    def log_batch(self, architecture: str, laboratory_name: str, experiment: str, era: int, epoch: int, batch: int, measurement: float):
        self.__write_lines(self.__entry_of(architecture, laboratory_name, experiment, "train", str(measurement), str(era), str(epoch), str(batch)))

    def log_test(self, architecture: str, laboratory_name: str, experiment: str, type_measurement: str, measurement: float):
        self.__write_lines(self.__entry_of(architecture, laboratory_name, experiment, "test-" + type_measurement, str(measurement)))

    def __write_lines(self, line: str):
        with open(self.path, 'a') as file:
            file.write(line + "\n")

    def __entry_of(self, architecture: str, laboratory_name: str, experiment: str, mode: str, value: str, era: str = "", epoch: str = "", batch: str = ""):
        return Delimiter.join([architecture, laboratory_name, experiment, era, epoch, batch, mode, value])

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
