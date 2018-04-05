# -*- coding: utf-8 -*-
import pandas as pd

from .variables import *


class DataHandler:
    def __init__(self):

        alpha_dict = {
            0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
            10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T",
            20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z"
        }
        alpha_len = len(alpha_dict)

        def _get_head_dict_key(index):
            key = str()

            if index < alpha_len:
                return alpha_dict[index]

            key_second = int(index / alpha_len) - 1
            key_first = index % alpha_len

            key += _get_head_dict_key(key_second)
            key += alpha_dict[key_first]

            return key

        self.file_name = DATA_READ
        self.rows_data = pd.read_csv(DATA_PATH + self.file_name)
        self.head_dict = {_get_head_dict_key(i): v for i, v in enumerate(self.rows_data)}
        self.y_data = list()
        self.erase_index_list = self.set_erase_index_list()
        self.data_dict = self.set_data_dict()

    # J : 연령, K : 성별, O : 주증상, AO : 수축혈압, AN : 의식, AP : 이완혈압, AQ : 맥박수, AR : 호흡수, AS : 체온
    def set_data_dict(self):
        data_dict = dict()

        header_list = ["J", "K", "O", "AN", "AO", "AP", "AQ", "AR", "AS"]

        for header in header_list:
            header_key = self.head_dict[header]

            # { J : [11, 50, ... , 45],  K : ['M', 'F', ... , 'M'], ... , AS : [36.6, 37.4, ... , 36.5] }
            data_dict[header] = list()
            for i, data in enumerate(self.rows_data[header_key]):
                if i not in self.erase_index_list:
                    data_dict[header].append(data)

        return data_dict

    # erase row WHERE "J" (연령) < 65, and WHERE "AN" (의식) == "-"
    def set_erase_index_list(self):

        # header keys 조건이 모두 만족 할 때
        def __condition__(header_list, condition):
            _header_keys = [self.head_dict[_i] for _i in header_list]

            _erase_index_dict = {_i: 0 for _i in range(len(self.rows_data[_header_keys[0]]))}

            for _header_key in _header_keys:
                for _index, _value in enumerate(self.rows_data[_header_key]):
                    _value = str(_value)

                    if condition == 0:
                        if _value == str(condition) or value == str(0.0) or value == "nan":
                            _erase_index_dict[_index] += 1
                    else:
                        if _value == str(condition):
                            _erase_index_dict[_index] += 1

            return _erase_index_dict, len(header_list)

        def __append__(_erase_index_dict, _num_match, _individual=False):
            for index, v in _erase_index_dict.items():
                if _individual and v >= _num_match:
                    if index not in erase_index_list:
                        erase_index_list.append(index)
                elif not _individual and v == _num_match:
                    if index not in erase_index_list:
                        erase_index_list.append(index)

        erase_index_list = list()

        header_key = self.head_dict["J"]
        for i, value in enumerate(self.rows_data[header_key]):
            if value < OLDER_AGE:
                erase_index_list.append(i)

        header_key = self.head_dict["AN"]
        for i, value in enumerate(self.rows_data[header_key]):
            if value == "-":
                erase_index_list.append(i)

        # AO : 수축혈압, AP : 이완혈압, AQ : 맥박수, AR : 호흡수 == 0 제외
        erase_index_dict, num_match = __condition__(header_list=["AO", "AP", "AQ", "AR"], condition=0)
        __append__(erase_index_dict, num_match)

        erase_index_dict, num_match = __condition__(header_list=["AO", "AP", "AQ", "AR"], condition=-1)
        __append__(erase_index_dict, num_match)

        # header_key = self.head_dict["AO"]
        # for i, value in enumerate(self.rows_data[header_key]):
        #     if value == 0 or value == -1:
        #         erase_index_list.append(i)
        #
        # header_key = self.head_dict["AS"]
        # for i, value in enumerate(self.rows_data[header_key]):
        #     if value == 99.9 or value == -1.0:
        #         erase_index_list.append(i)

        return sorted(list(set(erase_index_list)), reverse=True)

    # AE : 퇴실구분, BP : 입원 후 결과
    def set_labels(self):
        header_list = ["AE", "BP"]
        mortal_list = list()

        for header in header_list:
            header_key = self.head_dict[header]

            for i, value in enumerate(self.rows_data[header_key]):
                if value == "사망":
                    mortal_list.append(i)

        for i in range(len(self.rows_data[header_key])):
            if i in mortal_list:
                self.y_data.append([1])
            else:
                self.y_data.append([0])

        # erase exception on the labels
        for i in self.erase_index_list:
            self.y_data.pop(i)

    def free(self):
        del self.rows_data
        del self.head_dict
        del self.erase_index_list

    def counting_mortality(self, data):
        count = 0
        for i in data:
            if i == [1]:
                count += 1

        return count

    def show_data(self):

        for k, v in self.data_dict.items():
            print(k, len(v), v)

