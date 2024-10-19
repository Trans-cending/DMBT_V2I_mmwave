import os
import torch
from torch.utils.data import Dataset, DataLoader
from params import W, V, train_batch_size, test_batch_size, M, ori_M


class Testdataset_nfilter(Dataset):

    def __init__(self, scenario):
        self.test_sample_num = torch.load(os.path.join(r'Read_data_nfilter', scenario, r'test_sample_num.pkl'))
        self.test_unit2_location_combination = torch.load(os.path.join(r'Read_data_nfilter', scenario,
                                                                            r'test_unit2_location_combination.pkl'))
        self.test_unit2_gpsx_combination = torch.load(
            os.path.join(r'Read_data_nfilter', scenario, r'test_unit2_gpsx_combination.pkl'))
        self.test_unit2_gpsy_combination = torch.load(
            os.path.join(r'Read_data_nfilter', scenario, r'test_unit2_gpsy_combination.pkl'))
        self.test_unit1_beam_index_label_combination = torch.load(
            os.path.join(r'Read_data_nfilter', scenario, r'test_unit1_beam_index_combination.pkl'))
        self.test_unit1_mmwave_combination = torch.load(
            os.path.join(r'Read_data_nfilter', scenario, r'test_unit1_mmwave_combination.pkl'))

    def __len__(self):
        return self.test_sample_num

    def __getitem__(self, sample):
        test_unit2_location_root_set = self.test_unit2_location_combination[sample]
        test_unit1_gpsx_set = torch.zeros([W + V, 1])
        test_unit1_gpsy_set = torch.zeros([W + V, 1])
        for j in range(W):
            test_unit1_gpsx_set[j, :] = self.test_unit2_gpsx_combination[sample][j]
            test_unit1_gpsy_set[j, :] = self.test_unit2_gpsy_combination[sample][j]

        test_unit1_mmwave_set = torch.zeros([W + V, ori_M])
        for j in range(W):
            path_data = self.test_unit1_mmwave_combination[sample][j]
            file = open(path_data, "r")
            lines = file.readlines()
            mmwave_data = []
            for line in lines:
                line = line.strip()
                if line:
                    float_value = float(line)
                    mmwave_data.append(float_value)
            file.close()
            mmwave_data = torch.FloatTensor(mmwave_data)
            test_unit1_mmwave_set[j, :] = mmwave_data

        if M == 32:
            test_unit1_mmwave_set = test_unit1_mmwave_set[:, ::2] + test_unit1_mmwave_set[:, 1::2]

        test_beam_index_label_set = torch.zeros([V + 1, 1])
        for j in range(V + 1):
            test_beam_index_label_set[j, :] = int(self.test_unit1_beam_index_label_combination[sample][j] - 1)
        test_beam_index_label_set = test_beam_index_label_set.long()
        temp = tuple([test_unit2_location_root_set, test_unit1_gpsx_set, test_unit1_gpsy_set, test_unit1_mmwave_set,
                      test_beam_index_label_set])
        return temp


