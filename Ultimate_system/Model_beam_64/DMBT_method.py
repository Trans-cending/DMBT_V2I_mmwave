from sklearn.preprocessing import MinMaxScaler
import torch
import random
# import time
import numpy as np
import joblib
import torch.nn.functional as F
import os
import params



def extract_scenario(path):
    parts = path.rsplit('\\', 3)
    return parts[-1]


class DMBT_method():
    def Initialize_params(self, params_list):
        print("-----Initialize_params-----")
        self.N_r = params_list['N_r']
        self.b_th = params_list["b_th"]
        self.notx_l_th = params_list["notx_l_th"]
        self.k = params_list["k"]
        self.m = params_list["m"]
        self.Scenario = params_list["Scenario"]
        self.device = "cpu"
        self.params_list = params_list
        print("model on " + self.Scenario)

        self.hat_q_ori_list = []
        self.hat_q_list = []
        self.label_q_list = []
        self.trust_list = []
        self.top_1_power_list = []
        self.M = params.M

    def seed_everything(self, seed=params.random_state):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def Generate_dataset(self, scenario):
        print("-----Generate_dataset-----")
        test_loader = torch.load(os.path.join(r'Data_loader_nfilter',
                                                   scenario,r'test_loader_'+str(params.test_batch_size)+'.pkl'))
        self.test_loader = []
        for i, data in enumerate(test_loader):
            self.test_loader.append(data)


    def Generate_models(self, mm_model_list, loc_model_list, detect_model_list):
        print("-----Generate_models-----")
        mm_model_root = mm_model_list[-1]   # models formed in list for easier test if there are multiple models
        loc_model_root = loc_model_list[-1]
        detect_model_root = detect_model_list[-1]
        # model input
        self.MPBP = joblib.load(mm_model_root).to(self.device)
        self.MPBP.eval()
        self.RGUS = joblib.load(loc_model_root).to(self.device)
        self.RGUS.eval()
        self.LBP = joblib.load(detect_model_root).to(self.device)
        self.LBP.eval()

    def Simulation_process(self, MPBP_enable=1, RGBP_enable=1, error_mm=0, error_loc=0,
                           error_prob=0.1, noise_mm_factor=0.5, noise_loc_factor=0.15):
        print("-----Simulation_process-----")
        assert error_mm * error_loc != 1, "Assume error_mm and error_loc happen at different time. Low probability" \
                                          "that both happen meanwhile in practical use"
        assert (not MPBP_enable) * (not RGBP_enable) != 1, "MPBP_enable and RGBP_enable cannot be 0 meanwhile"
        assert self.N_r == int(self.N_r), "N_r should be int type"
        assert (error_prob >= 0 and error_prob <= 1), "error_prob should be within [0, 1]"

        self.seed_everything()
        self.MPBP_enable = MPBP_enable
        self.RGBP_enable = RGBP_enable
        self.error_mm = error_mm
        self.error_loc = error_loc
        self.error_prob = error_prob
        self.noise_mm_factor = 10 ** (noise_mm_factor / 10)  # transform dB to exact value
        self.noise_loc_factor = noise_loc_factor

        # min-max normalization
        scaler = MinMaxScaler()
        k = self.k
        m = self.m
        notx_l_th = self.notx_l_th
        U = params.W - 1
        kk = 0  # representing current sequence index
        initi = 1   # initialization at the very beginning
        times = 0
        replace = 0
        flag = 0
        times_set = max([1, self.N_r]) - 1
        observe_window = max([3, self.N_r+1]) + 1    # one period of DMBT method
        self.q_trained_list = []

        with torch.no_grad():
            # begin while
            while 1:
                for t in range(1, observe_window):
                    # time window: 1, 2, ..., max([3, self.N_r+1])
                    # at each moment, NPP conducts and ends, while BMA and BTC conducts and ends at max([1, self.N_r])-1
                    # initialize
                    if initi == 1:
                        data_0 = self.test_loader[0]
                        mmwave = data_0[3].clone().to(self.device)  # because the reference, using clone() instead
                        COS = mmwave.clone().to(self.device)
                        COS[:, U, :] = 0  # mask now mmwave
                        COS = COS.reshape([-1, params.M])
                        COS = torch.transpose(COS, dim0=0, dim1=1)
                        COS = torch.FloatTensor(scaler.fit_transform(COS.detach().cpu())).to(self.device)
                        COS = torch.transpose(COS, dim0=0, dim1=1)
                        COS = COS ** params.order_alpha
                        COS = COS.reshape([-1, U + 1, params.M])
                        mmwave_energy_1 = torch.argmax(COS[:, 0:U, :], dim=-1)
                        ORS = COS.clone().to(self.device)
                        ORS_other = COS.clone().to(self.device)
                        unit2_location = data_0[0][U-1][0]
                        gpsx = data_0[1].clone().to(self.device)[:, U, :]
                        gpsy = data_0[2].clone().to(self.device)[:, U, :]
                        now_gps = torch.concat((gpsx, gpsy), dim=-1)
                        unit2_location = np.asarray(unit2_location)
                        if not (unit2_location[0, 0] == -1. and unit2_location[0, 1] == -1.):
                            # add noise in location data
                            if error_loc:
                                rand = np.random.uniform(0, 1)
                                if rand < error_prob:
                                    unit2_location = np.abs(
                                        unit2_location + np.random.normal(loc=0, scale=noise_loc_factor,
                                                                          size=unit2_location.shape))

                                    unit2_location[unit2_location > 1] = 1
                                    unit2_location[unit2_location < 0] = 0

                            predict_loc = self.LBP(now_gps)
                            predict_loc = predict_loc.detach().cpu().numpy()
                            diviation_loc = np.sum(np.abs(predict_loc - unit2_location), axis=-1)
                            detect_Tx_index = np.argmin(diviation_loc)
                            if (diviation_loc[detect_Tx_index] >= notx_l_th):
                                now_locx = [0]
                                now_locy = [0]
                                now_locx = torch.as_tensor(now_locx, dtype=torch.float, device=self.device)
                                now_locy = torch.as_tensor(now_locy, dtype=torch.float, device=self.device)
                                now_loc = torch.concat((now_locx, now_locy), dim=-1)
                            else:
                                now_locx = unit2_location[detect_Tx_index][0]
                                now_locy = unit2_location[detect_Tx_index][1]
                                now_loc = torch.as_tensor((now_locx, now_locy), dtype=torch.float,
                                                          device=self.device).unsqueeze(0)

                        now_check_out = self.RGUS(now_loc, device=self.device)
                        hat_p_l = F.log_softmax(now_check_out, dim=-1).reshape([-1, self.M])
                        hat_q_l_ori = torch.argsort(hat_p_l, dim=-1, descending=True)
                        for j in range(U - 1):
                            self.q_trained_list.append(mmwave_energy_1[0, j])
                        q_1_0 = self.q_trained_list[-1]
                        initi = 0
                        break

                    if t == 1:
                        # save the first data sample index
                        kk0 = kk

                    if (self.N_r <= 1 and t == 2):
                        # NPP operates at each moment
                        combined_hat_p_m = self.MPBP.forward(COS)
                        combined_hat_p_m = combined_hat_p_m[:, U, :]  # extract now mmwave_seq energy predict
                        combined_hat_p_m = F.log_softmax(combined_hat_p_m, dim=-1)  # calculate probability
                        combined_hat_q_m = torch.argsort(combined_hat_p_m, descending=True, dim=-1)

                    # NPP operates at each moment, with some process of BTC
                    # update COS, ORS
                    if t >= 3:
                        combined_hat_p_m = self.MPBP.forward(COS)
                        combined_hat_p_m = combined_hat_p_m[:, U, :]      # extract now mmwave_seq energy predict
                        combined_hat_p_m = F.log_softmax(combined_hat_p_m, dim=-1)     # calculate probability
                        combined_hat_q_m = torch.argsort(combined_hat_p_m, descending=True, dim=-1)
                        bad_hat_p_m = self.MPBP.forward(ORS)[:, U, :]
                        bad_hat_q_m = torch.argsort(bad_hat_p_m, descending=True, dim=-1)
                        bad_hat_p_m_other = self.MPBP.forward(ORS_other)[:, U, :]
                        bad_hat_q_m_other = torch.argsort(bad_hat_p_m_other, descending=True, dim=-1)

                        if self.N_r <= 1:
                            if flag:
                                # 'Union' process
                                index_num = 0
                                total_num = m
                                hat_q_final = torch.empty([params.test_batch_size, total_num], device=self.device).long()
                                while 1:
                                    for i in range(params.M):
                                        equal = (hat_q_final[:, 0:index_num] == combined_hat_q_m[:, i]).any().item()
                                        if not equal:
                                            hat_q_final[:, index_num] = combined_hat_q_m[:, i]
                                            break
                                    index_num += 1
                                    if index_num == total_num:
                                        break

                                    for i in range(params.M):
                                        equal = (hat_q_final[:, 0:index_num] == hat_q_l_ori[:, i]).any().item()
                                        if not equal:
                                            hat_q_final[:, index_num] = hat_q_l_ori[:, i]
                                            break
                                    index_num += 1
                                    if index_num == total_num:
                                        break

                                    for i in range(params.M):
                                        equal = (hat_q_final[:, 0:index_num] == bad_hat_q_m[:, i]).any().item()
                                        if not equal:
                                            hat_q_final[:, index_num] = bad_hat_q_m[:, i]
                                            break
                                    index_num += 1
                                    if index_num == total_num:
                                        break
                                flag = 0

                            else:
                                hat_q_final = combined_hat_q_m[:, 0:k].clone().to(self.device)

                        elif self.N_r == 2:
                            if times == 0:
                                hat_q_final = combined_hat_q_m[:, 0:k].clone().to(self.device)
                            elif times > 0:
                                if times == times_set:
                                    index_num = 0
                                    total_num = m
                                    hat_q_final = torch.empty([params.test_batch_size, total_num], device=self.device).long()
                                    while 1:
                                        for i in range(params.M):
                                            equal = (hat_q_final[:, 0:index_num] == combined_hat_q_m[:, i]).any().item()
                                            if not equal:
                                                hat_q_final[:, index_num] = combined_hat_q_m[:, i]
                                                break
                                        index_num += 1
                                        if index_num == total_num:
                                            break

                                        for i in range(params.M):
                                            equal = (hat_q_final[:, 0:index_num] == bad_hat_q_m[:, i]).any().item()
                                            if not equal:
                                                hat_q_final[:, index_num] = bad_hat_q_m[:, i]
                                                break
                                        index_num += 1
                                        if index_num == total_num:
                                            break

                                else:
                                    hat_q_final = combined_hat_q_m[:, 0:m].clone().to(self.device)
                                times -= 1

                        elif self.N_r <= U + 1 and self.N_r > 2:
                            if times == 0:
                                hat_q_final = combined_hat_q_m[:, 0:k].clone().to(self.device)
                            elif times > 0:
                                if times == times_set:
                                    check_hat_p_m = self.MPBP.forward(CHS)[:, U, :]
                                    check_hat_q_m = torch.argsort(check_hat_p_m, descending=True, dim=-1)
                                    index_num = 0
                                    total_num = m
                                    hat_q_final = torch.empty([params.test_batch_size, total_num], device=self.device).long()
                                    while 1:
                                        for i in range(params.M):
                                            equal = (hat_q_final[:, 0:index_num] == combined_hat_q_m[:, i]).any().item()
                                            if not equal:
                                                hat_q_final[:, index_num] = combined_hat_q_m[:, i]
                                                break
                                        index_num += 1
                                        if index_num == total_num:
                                            break

                                        for i in range(params.M):
                                            equal = (hat_q_final[:, 0:index_num] == check_hat_q_m[:, i]).any().item()
                                            if not equal:
                                                hat_q_final[:, index_num] = check_hat_q_m[:, i]
                                                break
                                        index_num += 1
                                        if index_num == total_num:
                                            break

                                        for i in range(params.M):
                                            equal = (hat_q_final[:, 0:index_num] == bad_hat_q_m[:, i]).any().item()
                                            if not equal:
                                                hat_q_final[:, index_num] = bad_hat_q_m[:, i]
                                                break
                                        index_num += 1
                                        if index_num == total_num:
                                            break

                                else:
                                    hat_q_final = combined_hat_q_m[:, 0:m].clone().to(self.device)
                                times -= 1

                        else:
                            if times == 0:
                                hat_q_final = combined_hat_q_m[:, 0:k].clone().to(self.device)
                            elif times > 0:
                                if times == times_set:
                                    check_hat_p_m = self.MPBP.forward(CHS)[:, U, :]
                                    check_hat_q_m = torch.argsort(check_hat_p_m, descending=True, dim=-1)
                                    index_num = 0
                                    total_num = m
                                    hat_q_final = torch.empty([params.test_batch_size, total_num], device=self.device).long()
                                    while 1:    # ORS is equal to COS
                                        for i in range(params.M):
                                            equal = (hat_q_final[:, 0:index_num] == check_hat_q_m[:, i]).any().item()
                                            if not equal:
                                                hat_q_final[:, index_num] = check_hat_q_m[:, i]
                                                break
                                        index_num += 1
                                        if index_num == total_num:
                                            break

                                        for i in range(params.M):
                                            equal = (hat_q_final[:, 0:index_num] == bad_hat_q_m[:, i]).any().item()
                                            if not equal:
                                                hat_q_final[:, index_num] = bad_hat_q_m[:, i]
                                                break
                                        index_num += 1
                                        if index_num == total_num:
                                            break

                                else:
                                    hat_q_final = combined_hat_q_m[:, 0:m].clone().to(self.device)
                                times -= 1

                        # select to save results based on different condition
                        # *_ori denotes original set with cardinal M
                        if MPBP_enable == 1 and RGBP_enable == 1:
                            # set number of beams as |S_FIN|, representing proposed DMBT method
                            hat_q_condition = hat_q_final.clone().to(self.device)
                            hat_q_condition_ori = hat_q_final.clone().to(self.device)
                        elif MPBP_enable == 1 and RGBP_enable == 0:
                            # set number of beams as |S_FIN|, representing enhanced baseline 1
                            hat_q_condition = bad_hat_q_m_other[:, 0:hat_q_final.shape[-1]].clone().to(self.device)
                            hat_q_condition_ori = bad_hat_q_m_other.clone().to(self.device)
                            # set number pf beams as k
                            if self.N_r == 0:
                                # N_r == 0 has no practical meanings, representing results is for baseline 1
                                hat_q_condition = bad_hat_q_m_other[:, 0:k].clone().to(self.device)
                        elif MPBP_enable == 0 and RGBP_enable == 1:
                            # set number of beams as |S_FIN|, representing enhanced baseline 2
                            hat_q_condition = hat_q_l_ori[:, 0:hat_q_final.shape[-1]].clone().to(self.device)
                            hat_q_condition_ori = hat_q_l_ori.clone().to(self.device)
                            if self.N_r == 0:
                                # N_r == 0 has no practical meanings, representing results is for baseline 2
                                hat_q_condition = hat_q_l_ori[:, 0:k].clone().to(self.device)

                        hat_q_condition_ori = hat_q_condition_ori.long()
                        hat_q_condition = hat_q_condition.long()
                        self.hat_q_list.append(hat_q_condition.cpu().detach().cpu().numpy())
                        self.hat_q_ori_list.append(hat_q_condition_ori.cpu().detach().cpu().numpy())
                        self.trust_list.append(hat_q_condition.shape[-1])

                        # simulate partial beam training
                        # through the above operation, predicted beams subset is obtained
                        data_0 = self.test_loader[kk]
                        mmwave_energy_copy = data_0[3].clone().to(self.device)
                        mmwave_energy_copy = mmwave_energy_copy.reshape([-1, params.M])
                        mmwave_energy_copy = torch.transpose(mmwave_energy_copy, dim0=0, dim1=1)
                        mmwave_energy_copy = torch.FloatTensor(scaler.fit_transform(mmwave_energy_copy.detach().cpu())).to(self.device)
                        mmwave_energy_copy = torch.transpose(mmwave_energy_copy, dim0=0, dim1=1)
                        mmwave_energy_copy = mmwave_energy_copy.reshape([-1, U + 1, params.M])
                        self.label_q_list.append(data_0[4].item())
                        ori_now_mmwave = data_0[3].clone().to(self.device)[:, U, :].to(self.device)     # extract now RSP vector
                        now_mmwave = torch.zeros([params.test_batch_size, params.M], device=self.device)
                        now_mmwave_other = torch.zeros([params.test_batch_size, params.M], device=self.device)

                        # add noise for RSP vector
                        if error_mm:
                            rand = np.random.uniform(0, 1)
                            if rand < error_prob:
                                now_mmwave = torch.zeros([params.test_batch_size, params.M], device=self.device)
                                ori_now_mmwave_noise = ori_now_mmwave + torch.as_tensor(
                                    torch.randn([1, params.M]) * torch.max(ori_now_mmwave).item() / self.noise_mm_factor
                                    , device=self.device)
                                ori_now_mmwave_noise = torch.abs(ori_now_mmwave_noise)
                                # add noise under different conditions
                                for rows, cols in enumerate(hat_q_final):
                                    now_mmwave[rows, cols] = ori_now_mmwave_noise[rows, cols]
                                q_bar_opt_1 = torch.argmax(now_mmwave)
                                for rows, cols in enumerate(hat_q_condition):
                                    now_mmwave_other[rows, cols] = ori_now_mmwave_noise[rows, cols]
                                q_bar_opt_1_other = torch.argmax(now_mmwave_other)
                            else:
                                for rows, cols in enumerate(hat_q_final):
                                    now_mmwave[rows, cols] = ori_now_mmwave[rows, cols]
                                q_bar_opt_1 = torch.argmax(now_mmwave)
                                for rows, cols in enumerate(hat_q_condition):
                                    now_mmwave_other[rows, cols] = ori_now_mmwave[rows, cols]
                                q_bar_opt_1_other = torch.argmax(now_mmwave_other)
                        else:
                            for rows, cols in enumerate(hat_q_final):
                                now_mmwave[rows, cols] = ori_now_mmwave[rows, cols]
                            q_bar_opt_1 = torch.argmax(now_mmwave)
                            for rows, cols in enumerate(hat_q_condition):
                                now_mmwave_other[rows, cols] = ori_now_mmwave[rows, cols]
                            q_bar_opt_1_other = torch.argmax(now_mmwave_other)

                        # save top_1_power and best beam index
                        # note that top_1_power comes from noise free RSP instead of noisy RSP
                        if MPBP_enable == 1 and RGBP_enable == 1:
                            self.top_1_power_list.append(
                                mmwave_energy_copy[:, U].reshape(self.M)[q_bar_opt_1].item())
                            self.q_trained_list.append(q_bar_opt_1.item())
                        else:
                            self.top_1_power_list.append(
                                mmwave_energy_copy[:, U].reshape(self.M)[q_bar_opt_1_other].item())
                            self.q_trained_list.append(q_bar_opt_1_other.item())

                        now_mmwave = torch.transpose(now_mmwave, dim0=0, dim1=1)
                        now_mmwave = torch.FloatTensor(scaler.fit_transform(now_mmwave.detach().cpu())).to(self.device)
                        now_mmwave = torch.transpose(now_mmwave, dim0=0, dim1=1)
                        now_mmwave = now_mmwave.reshape(
                            [params.test_batch_size, 1,
                             params.M]) ** params.order_alpha  # normalize must be before the concat

                        now_mmwave_other = torch.transpose(now_mmwave_other, dim0=0, dim1=1)
                        now_mmwave_other = torch.FloatTensor(scaler.fit_transform(now_mmwave_other.detach().cpu())).to(self.device)
                        now_mmwave_other = torch.transpose(now_mmwave_other, dim0=0, dim1=1)
                        now_mmwave_other = now_mmwave_other.reshape(
                            [params.test_batch_size, 1,
                             params.M]) ** params.order_alpha  # normalize must be before the concat

                        temp1 = torch.concat((COS[:, 1:U, :], now_mmwave), dim=1)
                        temp2 = torch.concat((ORS[:, 1:U, :], now_mmwave), dim=1)
                        # for baseline 1 and enhanced baseline 1
                        temp3 = torch.concat((ORS_other[:, 1:U, :], now_mmwave_other), dim=1)

                        # update sequence
                        COS[:, 0:U, :] = temp1
                        COS[:, U, :] = 0

                        ORS[:, 0:U, :] = temp2
                        ORS[:, U, :] = 0

                        ORS_other[:, 0:U, :] = temp3
                        ORS_other[:, U, :] = 0

                        if (self.N_r <= 1 and replace):
                            # if N_r == 1 and misalignment detected, COS can be modified immediately
                            COS[:, U - 1, :] = hat_p_l
                            replace = 0

                        if t == 3:
                            initi_mmwave_energy = COS.clone().to(self.device)
                            q_1_0 = torch.as_tensor(q_bar_opt_1, device=self.device)

                        kk += 1     # update data sample index

                    # RGUS and LBP operate
                    if (self.N_r <= 1 and t == 2) or (self.N_r >= 2 and t == observe_window - 1):
                        data_1 = self.test_loader[kk0]
                        unit2_location = data_1[0][U][0]
                        gpsx = data_1[1].clone().to(self.device)[:, U, :]
                        gpsy = data_1[2].clone().to(self.device)[:, U, :]
                        now_gps = torch.concat((gpsx, gpsy), dim=-1)
                        unit2_location = np.asarray(unit2_location)
                        if not (unit2_location[0, 0] == -1. and unit2_location[0, 1] == -1.):
                            # add noise in location data
                            if error_loc:
                                rand = np.random.uniform(0, 1)
                                if rand < error_prob:
                                    unit2_location = np.abs(
                                        unit2_location + np.random.normal(loc=0, scale=noise_loc_factor,
                                                                          size=unit2_location.shape))

                                    unit2_location[unit2_location > 1] = 1
                                    unit2_location[unit2_location < 0] = 0

                            predict_loc = self.LBP(now_gps)
                            predict_loc = predict_loc.detach().cpu().numpy()
                            diviation_loc = np.sum(np.abs(predict_loc - unit2_location), axis=-1)
                            detect_Tx_index = np.argmin(diviation_loc)
                            if (diviation_loc[detect_Tx_index] >= notx_l_th):
                                now_locx = [0]
                                now_locy = [0]
                                now_locx = torch.as_tensor(now_locx, dtype=torch.float, device=self.device)
                                now_locy = torch.as_tensor(now_locy, dtype=torch.float, device=self.device)
                                now_loc = torch.concat((now_locx, now_locy), dim=-1)
                            else:
                                now_locx = unit2_location[detect_Tx_index][0]
                                now_locy = unit2_location[detect_Tx_index][1]
                                now_loc = torch.as_tensor((now_locx, now_locy), dtype=torch.float,
                                                          device=self.device).unsqueeze(0)

                        now_check_out = self.RGUS(now_loc, device=self.device)
                        hat_p_l = F.log_softmax(now_check_out, dim=-1).reshape([-1, self.M])
                        hat_p_l_ori = hat_p_l.clone().to(self.device)
                        hat_q_l_1 = torch.argmax(hat_p_l_ori, dim=-1)
                        hat_q_l_ori = torch.argsort(hat_p_l, dim=-1, descending=True)
                        hat_q_l = hat_q_l_ori[:, 0:m].clone().to(self.device)

                        # N_r > 1, can only obtain q_1_0
                        if self.N_r >= 2:
                            if_div = torch.sum(torch.abs(hat_q_l_1 - q_1_0)) >= self.b_th
                        elif self.N_r <= 1:
                            if_div = torch.sum(
                                torch.min(torch.abs(torch.subtract(hat_q_l[:, 0:1], combined_hat_q_m[:, 0:1])))) >= self.b_th

                        # 'Iteration' process
                        if if_div:
                            times = times_set
                            flag = 1
                            replace = 1
                            hat_p_l = torch.transpose(hat_p_l, dim0=0, dim1=1)
                            hat_p_l = torch.FloatTensor(scaler.fit_transform(hat_p_l.detach().cpu())).to(self.device)
                            hat_p_l = torch.transpose(hat_p_l, dim0=0, dim1=1)
                            hat_p_l = hat_p_l.reshape(
                                [params.test_batch_size, 1,
                                 params.M]) ** params.order_beta
                            # form CHS and update COS
                            if self.N_r >= 2:
                                again_times = self.N_r - 2      # again_times: 0, 1, ...
                                initi_mmwave_energy[:, U - 1, :] = hat_p_l
                                # iteration correction
                                for i in range(0, again_times):
                                    hat_p_l = self.MPBP.forward(initi_mmwave_energy)
                                    hat_p_l = torch.log_softmax(hat_p_l, dim=-1)
                                    hat_p_l = hat_p_l[:, U, :]
                                    hat_p_l_ori = hat_p_l.clone().to(self.device)
                                    hat_q_l_ori = torch.argsort(hat_p_l, dim=-1, descending=True)
                                    hat_p_l = torch.transpose(hat_p_l, dim0=0, dim1=1)
                                    hat_p_l = torch.FloatTensor(scaler.fit_transform(hat_p_l.detach().cpu())).to(self.device)
                                    hat_p_l = torch.transpose(hat_p_l, dim0=0, dim1=1)
                                    hat_p_l = hat_p_l.reshape(
                                        [params.test_batch_size, 1,
                                         params.M]) ** params.order_beta  # normalize must be before the concat
                                    temp4 = torch.concat((initi_mmwave_energy[:, 1: U, :], hat_p_l), dim=1)
                                    # update CHS
                                    initi_mmwave_energy[:, 0:U, :] = temp4
                                    initi_mmwave_energy[:, U, :] = 0
                                CHS = initi_mmwave_energy.clone().to(self.device)
                                # update COS
                                COS[:, U-1, :] = hat_p_l

                    if kk == len(self.test_loader):
                        print("test ends: " + str(kk))
                        self.kk = kk
                        return

    def Evaluation_process(self, n_list=None, write_csv=0):
        print("-----Evaluation_process-----")

        # check
        assert self.kk == self.hat_q_ori_list.__len__() and self.kk == self.label_q_list.__len__() and \
               self.kk == self.trust_list.__len__() and self.kk == self.top_1_power_list.__len__() and \
               self.kk == self.q_trained_list.__len__() - params.W + 2, \
            print("check code, length error",
                  {"self.hat_q_ori_list": self.hat_q_ori_list.__len__(),
                   "self.q_trained_list": self.q_trained_list.__len__() - params.W + 2,
                   "self.label_q_list": self.label_q_list.__len__(),
                   "self.trust_list": self.trust_list.__len__(),
                   "self.top_1_power_list": self.top_1_power_list.__len__()
                   })

        top_n_acc_list = []
        for idx, n in enumerate(n_list):
            top_n_acc = 0
            for i in range(len(self.hat_q_ori_list)):
                labels_now = np.asarray(self.label_q_list[i])
                tiled_top_n_labels = labels_now.repeat(n).reshape([1, -1])
                top_n_acc += int(np.sum(self.hat_q_ori_list[i][:, 0:n] == tiled_top_n_labels) > 0)
            top_n_acc_list.append(top_n_acc / len(self.hat_q_ori_list))

        top_km_acc = 0
        for i in range(len(self.hat_q_list)):
            labels_now = np.asarray(self.label_q_list[i])
            tiled_top_km_labels = labels_now.repeat(len(self.hat_q_list[i][0, :])).reshape([1, -1])
            top_km_acc += int(np.sum(self.hat_q_list[i][0, :] == tiled_top_km_labels) > 0)
        top_km_acc = top_km_acc / len(self.hat_q_ori_list)

        trust_list = np.asarray(self.trust_list)
        par_k_times = np.sum(trust_list == self.k)
        par_m_times = np.sum(trust_list == self.m)
        ave_top_1_power = np.sum(np.asarray(self.top_1_power_list)) / len(self.top_1_power_list)
        ave_overhead = np.sum(trust_list) / len(self.hat_q_list)
        print(
            f'''
                params list                                 {self.params_list}
                error_mm_prob                                   {self.error_mm * self.error_prob}
                error_loc_prob                                  {self.error_loc * self.error_prob}
                error_mm_factor                             {self.noise_mm_factor}
                error_loc_factor                           {self.noise_loc_factor}
                MPBP_enable                                 {self.MPBP_enable}
                RGBP_enable                                 {self.RGBP_enable}
                training selected m beams times                 {par_m_times}
                training selected k beams times                 {par_k_times}
                ave Top 1 Power               {ave_top_1_power}
                Average beam training overhead      {ave_overhead}
                '''
        )
        return top_n_acc_list, top_km_acc, ave_top_1_power, ave_overhead
