#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""csv_data.py
Created by Dongmin Kim at 2023-08-31

This module does stuff.
"""
import csv
import numpy as np
import h5py
from pandas import read_csv


class CSVData:
    def __init__(
        self,
        log_path=None,
        category_name=None,
        write_to_disk=True,
        append=False,
    ):
        self.write_to_disk = write_to_disk
        if not write_to_disk:
            return
        self.write_to_disk = write_to_disk
        self.category_name = category_name
        self.file_name = f"{log_path}/{category_name}.csv"
        mode = "a" if append else "w"
        self.f = open(self.file_name, mode, newline="")
        self.writer = csv.writer(self.f, delimiter="\t")
        self.data_value_list = list()

    def cleanup(self):
        if not self.write_to_disk:
            return

        self.f.flush()
        self.f.close()

    def append_data_value(self, data_value):
        if not self.write_to_disk:
            return

        self.data_value_list.append(np.copy(data_value))

    def flush_data(self):
        if not self.write_to_disk:
            return

        self.writer.writerows(self.data_value_list)
        self.data_value_list.clear()
        self.f.flush()


class ConvertCSVtoHDF5:
    def __init__(self, file_name=None, csv_data_list=None, write_to_disk=True):
        self.write_to_disk = write_to_disk

        if not write_to_disk:
            return

        self.file_name = file_name
        self.csv_data_list = csv_data_list

    def real_convert(self, file_names, category_names):
        with h5py.File(self.file_name, "w", locking=False) as hf:
            for file_name, category_name in zip(file_names, category_names):
                with open(file_name, "r") as f:
                    data = read_csv(f, header=None, delimiter="\t")

                d = hf.create_dataset(category_name, data=data)
                # os.remove(csv_data.file_name)

        print("h5py convert done")

    def convert(self):
        if not self.write_to_disk:
            return

        file_names = [csv_data.file_name for csv_data in self.csv_data_list]
        category_names = [csv_data.category_name for csv_data in self.csv_data_list]

        self.real_convert(file_names, category_names)
