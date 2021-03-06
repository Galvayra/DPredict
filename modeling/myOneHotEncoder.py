
MIN_SCALING = 0.1


class MyOneHotEncoder:
    def __init__(self):
        self.vector_dict = dict()

    # J : 연령, K : 성별, O : 주증상, AN : 의식, AO : 수축혈압, AP : 이완혈압, AQ : 맥박수, AR : 호흡수, AS : 체온
    def encoding(self, data_dict):
        def _set_scalar_dict():
            scalar_dict = dict()
            scalar_list = sorted(list(set(vector_list)))

            scalar_dict["min"] = scalar_list[0]
            scalar_dict["max"] = scalar_list[-1]
            scalar_dict["div"] = float(scalar_dict["max"] - scalar_dict["min"])

            return scalar_dict

        def _set_class_dict():
            class_dict = dict()

            for v in vector_list:
                if v not in class_dict:
                    class_dict[v] = 0

                class_dict[v] += 1

            return class_dict

        def _set_symptom_dict():
            symptom_dict = dict()

            for line in vector_list:
                line = line.strip().lower()
                if line.endswith(";"):
                    line = line[:-1]

                for symptom in line.split(";"):
                    symptom = symptom.strip()

                    if symptom not in symptom_dict:
                        symptom_dict[symptom] = 0

                    symptom_dict[symptom] += 1

            return symptom_dict

        for k, vector_list in data_dict.items():

            # key : 성별
            if k == "K":
                self.vector_dict[k] = _set_class_dict()
            # key : 주증상
            elif k == "O":
                self.vector_dict[k] = _set_symptom_dict()
                # print(len(self.vector_dict[k]))
            # key : 의식
            elif k == "AN":
                self.vector_dict[k] = _set_class_dict()
            # scalar vector
            else:
                self.vector_dict[k] = _set_scalar_dict()

    def fit(self, data_dict, data_count):
        def _init_x_data():
            _x_data = list()

            # set X(number of rows) using rows_data
            # array dimension = X * Y(number of data)
            for _i in range(data_count):
                _x_data.append(list())

            return _x_data

        def _make_vector_from_class():
            for c in class_list:
                if c == value:
                    x_data[i].append(1.0)
                else:
                    x_data[i].append(0.0)

        def _make_vector_from_word():
            line = value.strip().lower()
            if line.endswith(";"):
                line = line[:-1]
            symptom = [symptom.strip() for symptom in line.split(";")]

            for c in class_list:
                if c in symptom:
                    x_data[i].append(1.0)
                else:
                    x_data[i].append(0.0)

        x_data = _init_x_data()

        for k in sorted(data_dict.keys()):
            encode_dict = self.vector_dict[k]

            # key : 성별
            if k == "K":
                class_list = encode_dict.keys()
                for i, value in enumerate(data_dict[k]):
                    _make_vector_from_class()
            # key : 주증상
            elif k == "O":
                # pass
                class_list = encode_dict.keys()
                for i, value in enumerate(data_dict[k]):
                    _make_vector_from_word()
                    # _make_vector_from_class()
                    # print(x_data[i])
            # key : 의식
            elif k == "AN":
                class_list = encode_dict.keys()
                for i, value in enumerate(data_dict[k]):
                    _make_vector_from_class()
            # scalar vector
            else:
                minimum = encode_dict["min"]
                maximum = encode_dict["max"]
                division = encode_dict["div"]

                for i, value in enumerate(data_dict[k]):
                    # exception
                    if value <= 0 and value <= minimum:
                        value = float(0)
                    else:
                        if value > maximum:
                            value = float(1)
                        elif value < minimum:
                            value = MIN_SCALING/(division + MIN_SCALING)
                        # normalization
                        else:
                            value = (value - minimum + MIN_SCALING)/(division + MIN_SCALING)

                    x_data[i].append(value)

        return x_data
