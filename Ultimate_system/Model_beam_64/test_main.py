import DMBT_method as Tc
import os
import numpy as np
import pandas as pd

def demo_class(scenario, params_list, mm_model_list, loc_model_list, detect_model_list, if_mm,
               if_loc, if_error_mm, if_error_loc, n_list, write_csv, mm_factor=0.5, loc_factor=0.15, prob=0.):
    tc_obj = Tc.DMBT_method()
    tc_obj.Initialize_params(params_list)
    tc_obj.Generate_dataset(scenario=scenario)
    tc_obj.Generate_models(mm_model_list=mm_model_list,
                           loc_model_list=loc_model_list,
                           detect_model_list=detect_model_list)
    tc_obj.Simulation_process(MPBP_enable=if_mm, RGBP_enable=if_loc, error_mm=if_error_mm, error_loc=if_error_loc,
                              noise_mm_factor=mm_factor, noise_loc_factor=loc_factor, error_prob=prob)
    item = tc_obj.Evaluation_process(n_list=n_list, write_csv=write_csv)
    return item


def error_mm_test(N_r=1, prob=0.2):
    if not os.path.exists("Error_correction_comparison_mm"):
        os.makedirs("Error_correction_comparison_mm")

    results_dict = {"Normal case": {"Scenario 3": {"top3_acc": 0,
                                                   "ave_overhead": 0,
                                                   "ave_power": 0},
                                    "Scenario 4": {"top3_acc": 0,
                                                   "ave_overhead": 0,
                                                   "ave_power": 0},
                                    "Scenario 5": {"top3_acc": 0,
                                                   "ave_overhead": 0,
                                                   "ave_power": 0},
                                    "Scenario 8": {"top3_acc": 0,
                                                   "ave_overhead": 0,
                                                   "ave_power": 0},
                                    },
                    "Error case": {"Scenario 3": {"top3_acc": 0,
                                                  "ave_overhead": 0,
                                                  "ave_power": 0},
                                   "Scenario 4": {"top3_acc": 0,
                                                  "ave_overhead": 0,
                                                  "ave_power": 0},
                                   "Scenario 5": {"top3_acc": 0,
                                                  "ave_overhead": 0,
                                                  "ave_power": 0},
                                   "Scenario 8": {"top3_acc": 0,
                                                  "ave_overhead": 0,
                                                  "ave_power": 0},
                                   },
                    "Correct case": {"Scenario 3": {"top3_acc": 0,
                                                    "ave_overhead": 0,
                                                    "ave_power": 0},
                                     "Scenario 4": {"top3_acc": 0,
                                                    "ave_overhead": 0,
                                                    "ave_power": 0},
                                     "Scenario 5": {"top3_acc": 0,
                                                    "ave_overhead": 0,
                                                    "ave_power": 0},
                                     "Scenario 8": {"top3_acc": 0,
                                                    "ave_overhead": 0,
                                                    "ave_power": 0},
                                     }, }


    # model on S8
    scenario = r'Scenario8'
    mm_model_root_0 = r'..\\..\\Mmwave_aided_model\\Model_beam_64\\Model_on_S8\\mm_GRU_BP_21.joblib'
    loc_model_root = r'..\\..\\Location_aid_model\\Model_beam_64\\Model_on_S8\\loc_RFF_BM_65.joblib'
    detect_model_root = r'..\\..\\Candidate_detect\\Model_on_S8\\User_detect_BP_31.joblib'

    params_list = {"N_r": N_r,
                   "b_th": 5,
                   "notx_l_th": 0.5,
                   "k": 6,
                   "m": 9,
                   "Scenario": "Scenario 8"}
    mm_model_list = [mm_model_root_0]
    loc_model_list = [loc_model_root]
    detect_model_list = [detect_model_root]
    if_mm = 1
    if_loc = 0
    if_error_mm = 1
    if_error_loc = 0
    write_csv = 0
    n_list = [1, 3, 5, 6, 7]
    item = demo_class(scenario, params_list, mm_model_list, loc_model_list, detect_model_list,
                      if_mm,
                      if_loc,
                      if_error_mm, if_error_loc, n_list, write_csv, mm_factor, loc_factor, prob)
    results_dict["Error case"]["Scenario 8"]["top3_acc"] = item[0][1]
    results_dict["Error case"]["Scenario 8"]["ave_overhead"] = item[-1]
    results_dict["Error case"]["Scenario 8"]["ave_power"] = item[-2]

    mm_model_list = [mm_model_root_0]
    loc_model_list = [loc_model_root]
    detect_model_list = [detect_model_root]
    if_mm = 1
    if_loc = 1
    if_error_mm = 1
    if_error_loc = 0
    write_csv = 0
    n_list = [1, 3, 5, 6, 7]
    item = demo_class(scenario, params_list, mm_model_list, loc_model_list, detect_model_list,
                      if_mm,
                      if_loc,
                      if_error_mm, if_error_loc, n_list, write_csv, mm_factor, loc_factor, prob)
    results_dict["Correct case"]["Scenario 8"]["top3_acc"] = item[0][1]
    results_dict["Correct case"]["Scenario 8"]["ave_overhead"] = item[-1]
    results_dict["Correct case"]["Scenario 8"]["ave_power"] = item[-2]

    # Model on S3
    scenario = r'Scenario3'
    mm_model_root_0 = r'..\\..\\Mmwave_aided_model\\Model_beam_64\\Model_on_S3\\mm_GRU_BP_25.joblib'
    loc_model_root = r'..\\..\\Location_aid_model\\Model_beam_64\\Model_on_S3\\loc_RFF_BM_52.joblib'
    detect_model_root = r'..\\..\\Candidate_detect\\Model_on_S3\\User_detect_BP_112.joblib'

    params_list = {"N_r": N_r,
                   "b_th": 5,
                   "notx_l_th": 0.5,
                   "k": 8,
                   "m": 12,
                   "Scenario": "Scenario 3"}
    mm_model_list = [mm_model_root_0]
    loc_model_list = [loc_model_root]
    detect_model_list = [detect_model_root]
    if_mm = 1
    if_loc = 0
    if_error_mm = 1
    if_error_loc = 0
    write_csv = 0
    n_list = [1, 3, 5, 6, 7]
    item = demo_class(scenario, params_list, mm_model_list, loc_model_list, detect_model_list,
                      if_mm,
                      if_loc,
                      if_error_mm, if_error_loc, n_list, write_csv, mm_factor, loc_factor, prob)
    results_dict["Error case"]["Scenario 3"]["top3_acc"] = item[0][1]
    results_dict["Error case"]["Scenario 3"]["ave_overhead"] = item[-1]
    results_dict["Error case"]["Scenario 3"]["ave_power"] = item[-2]

    mm_model_list = [mm_model_root_0]
    loc_model_list = [loc_model_root]
    detect_model_list = [detect_model_root]
    if_mm = 1
    if_loc = 1
    if_error_mm = 1
    if_error_loc = 0
    write_csv = 0
    n_list = [1, 3, 5, 6, 7]
    item = demo_class(scenario, params_list, mm_model_list, loc_model_list, detect_model_list,
                      if_mm,
                      if_loc,
                      if_error_mm, if_error_loc, n_list, write_csv, mm_factor, loc_factor, prob)
    results_dict["Correct case"]["Scenario 3"]["top3_acc"] = item[0][1]
    results_dict["Correct case"]["Scenario 3"]["ave_overhead"] = item[-1]
    results_dict["Correct case"]["Scenario 3"]["ave_power"] = item[-2]



    # Model on S4
    scenario = r'Scenario4'
    mm_model_root_0 = r'..\\..\\Mmwave_aided_model\\Model_beam_64\\Model_on_S4\\mm_GRU_BP_20.joblib'
    loc_model_root = r'..\\..\\Location_aid_model\\Model_beam_64\\Model_on_S4\\loc_RFF_BM_10.joblib'
    detect_model_root = r'..\\..\\Candidate_detect\\Model_on_S4\\User_detect_BP_110.joblib'

    params_list = {"N_r": N_r,
                   "b_th": 5,
                   "notx_l_th": 0.5,
                   "k": 8,
                   "m": 15,
                   "Scenario": "Scenario 4"}
    mm_model_list = [mm_model_root_0]
    loc_model_list = [loc_model_root]
    detect_model_list = [detect_model_root]
    if_mm = 1
    if_loc = 0
    if_error_mm = 1
    write_csv = 0
    n_list = [1, 3, 5, 6, 7]

    item = demo_class(scenario, params_list, mm_model_list, loc_model_list, detect_model_list,
                      if_mm,
                      if_loc,
                      if_error_mm, if_error_loc, n_list, write_csv, mm_factor, loc_factor, prob)
    results_dict["Error case"]["Scenario 4"]["top3_acc"] = item[0][1]
    results_dict["Error case"]["Scenario 4"]["ave_overhead"] = item[-1]
    results_dict["Error case"]["Scenario 4"]["ave_power"] = item[-2]

    mm_model_list = [mm_model_root_0]
    loc_model_list = [loc_model_root]
    detect_model_list = [detect_model_root]
    if_mm = 1
    if_loc = 1
    if_error_mm = 1
    if_error_loc = 0
    write_csv = 0
    n_list = [1, 3, 5, 6, 7]
    item = demo_class(scenario, params_list, mm_model_list, loc_model_list, detect_model_list,
                      if_mm,
                      if_loc,
                      if_error_mm, if_error_loc, n_list, write_csv, mm_factor, loc_factor, prob)
    results_dict["Correct case"]["Scenario 4"]["top3_acc"] = item[0][1]
    results_dict["Correct case"]["Scenario 4"]["ave_overhead"] = item[-1]
    results_dict["Correct case"]["Scenario 4"]["ave_power"] = item[-2]

    # Model on S5
    scenario = r'Scenario5'
    mm_model_root_0 = r'..\\..\\Mmwave_aided_model\\Model_beam_64\\Model_on_S5\\mm_GRU_BP_27.joblib'
    loc_model_root = r'..\\..\\Location_aid_model\\Model_beam_64\\Model_on_S5\\loc_RFF_BM_11.joblib'
    detect_model_root = r'..\\..\\Candidate_detect\\Model_on_S5\\User_detect_BP_11.joblib'

    params_list = {"N_r": N_r,
                   "b_th": 5,
                   "notx_l_th": 0.5,
                   "k": 6,
                   "m": 9,
                   "Scenario": "Scenario 5"}
    mm_model_list = [mm_model_root_0]
    loc_model_list = [loc_model_root]
    detect_model_list = [detect_model_root]
    if_mm = 1
    if_loc = 0
    if_error_mm = 1
    if_error_loc = 0
    write_csv = 0
    n_list = [1, 3, 5, 6, 7]
    item = demo_class(scenario, params_list, mm_model_list, loc_model_list, detect_model_list,
                      if_mm,
                      if_loc,
                      if_error_mm, if_error_loc, n_list, write_csv, mm_factor, loc_factor, prob)
    results_dict["Error case"]["Scenario 5"]["top3_acc"] = item[0][1]
    results_dict["Error case"]["Scenario 5"]["ave_overhead"] = item[-1]
    results_dict["Error case"]["Scenario 5"]["ave_power"] = item[-2]

    mm_model_list = [mm_model_root_0]
    loc_model_list = [loc_model_root]
    detect_model_list = [detect_model_root]
    if_mm = 1
    if_loc = 1
    if_error_mm = 1
    if_error_loc = 0
    write_csv = 0
    n_list = [1, 3, 5, 6, 7]
    item = demo_class(scenario, params_list, mm_model_list, loc_model_list, detect_model_list,
                      if_mm,
                      if_loc,
                      if_error_mm, if_error_loc, n_list, write_csv, mm_factor, loc_factor, prob)
    results_dict["Correct case"]["Scenario 5"]["top3_acc"] = item[0][1]
    results_dict["Correct case"]["Scenario 5"]["ave_overhead"] = item[-1]
    results_dict["Correct case"]["Scenario 5"]["ave_power"] = item[-2]
    return results_dict


def error_loc_test(N_r=1, prob=0.2):
    if not os.path.exists("Error_correction_comparison_loc"):
        os.makedirs("Error_correction_comparison_loc")

    # error happens
    results_dict = {"Normal case": {"Scenario 3": {"top3_acc": 0,
                                                   "ave_overhead": 0,
                                                   "ave_power": 0},
                                    "Scenario 4": {"top3_acc": 0,
                                                   "ave_overhead": 0,
                                                   "ave_power": 0},
                                    "Scenario 5": {"top3_acc": 0,
                                                   "ave_overhead": 0,
                                                   "ave_power": 0},
                                    "Scenario 8": {"top3_acc": 0,
                                                   "ave_overhead": 0,
                                                   "ave_power": 0},
                                    },
                    "Error case": {"Scenario 3": {"top3_acc": 0,
                                                  "ave_overhead": 0,
                                                  "ave_power": 0},
                                   "Scenario 4": {"top3_acc": 0,
                                                  "ave_overhead": 0,
                                                  "ave_power": 0},
                                   "Scenario 5": {"top3_acc": 0,
                                                  "ave_overhead": 0,
                                                  "ave_power": 0},
                                   "Scenario 8": {"top3_acc": 0,
                                                  "ave_overhead": 0,
                                                  "ave_power": 0},
                                   },
                    "Correct case": {"Scenario 3": {"top3_acc": 0,
                                                    "ave_overhead": 0,
                                                    "ave_power": 0},
                                     "Scenario 4": {"top3_acc": 0,
                                                    "ave_overhead": 0,
                                                    "ave_power": 0},
                                     "Scenario 5": {"top3_acc": 0,
                                                    "ave_overhead": 0,
                                                    "ave_power": 0},
                                     "Scenario 8": {"top3_acc": 0,
                                                    "ave_overhead": 0,
                                                    "ave_power": 0},
                                     }, }

    # model on S8
    scenario = r'Scenario8'
    mm_model_root_0 = r'..\\..\\Mmwave_aided_model\\Model_beam_64\\Model_on_S8\\mm_GRU_BP_21.joblib'
    loc_model_root = r'..\\..\\Location_aid_model\\Model_beam_64\\Model_on_S8\\loc_RFF_BM_65.joblib'
    detect_model_root = r'..\\..\\Candidate_detect\\Model_on_S8\\User_detect_BP_31.joblib'

    params_list = {"N_r": N_r,
                   "b_th": 5,
                   "notx_l_th": 0.5,
                   "k": 6,
                   "m": 9,
                   "Scenario": "Scenario 8"}
    mm_model_list = [mm_model_root_0]
    loc_model_list = [loc_model_root]
    detect_model_list = [detect_model_root]
    if_mm = 0
    if_loc = 1
    if_error_mm = 0
    if_error_loc = 1
    write_csv = 0
    n_list = [1, 3, 5, 6, 7]
    item = demo_class(scenario, params_list, mm_model_list, loc_model_list, detect_model_list,
                       if_mm,
                       if_loc,
                       if_error_mm, if_error_loc, n_list, write_csv, mm_factor, loc_factor, prob)
    results_dict["Error case"]["Scenario 8"]["top3_acc"] = item[0][1]
    results_dict["Error case"]["Scenario 8"]["ave_overhead"] = item[-1]
    results_dict["Error case"]["Scenario 8"]["ave_power"] = item[-2]

    mm_model_list = [mm_model_root_0]
    loc_model_list = [loc_model_root]
    detect_model_list = [detect_model_root]
    if_mm = 1
    if_loc = 1
    if_error_mm = 0
    if_error_loc = 1
    write_csv = 0
    n_list = [1, 3, 5, 6, 7]
    item = demo_class(scenario, params_list, mm_model_list, loc_model_list, detect_model_list,
                       if_mm,
                       if_loc,
                       if_error_mm, if_error_loc, n_list, write_csv, mm_factor, loc_factor, prob)
    results_dict["Correct case"]["Scenario 8"]["top3_acc"] = item[0][1]
    results_dict["Correct case"]["Scenario 8"]["ave_overhead"] = item[-1]
    results_dict["Correct case"]["Scenario 8"]["ave_power"] = item[-2]

    # Model on S3
    scenario = r'Scenario3'
    mm_model_root_0 = r'..\\..\\Mmwave_aided_model\\Model_beam_64\\Model_on_S3\\mm_GRU_BP_25.joblib'
    loc_model_root = r'..\\..\\Location_aid_model\\Model_beam_64\\Model_on_S3\\loc_RFF_BM_52.joblib'
    detect_model_root = r'..\\..\\Candidate_detect\\Model_on_S3\\User_detect_BP_112.joblib'

    params_list = {"N_r": N_r,
                   "b_th": 5,
                   "notx_l_th": 0.5,
                   "k": 8,
                   "m": 12,
                   "Scenario": "Scenario 3"}
    mm_model_list = [mm_model_root_0]
    loc_model_list = [loc_model_root]
    detect_model_list = [detect_model_root]
    if_mm = 0
    if_loc = 1
    if_error_mm = 0
    if_error_loc = 1
    write_csv = 0
    n_list = [1, 3, 5, 6, 7]
    item = demo_class(scenario, params_list, mm_model_list, loc_model_list, detect_model_list,
                       if_mm,
                       if_loc,
                       if_error_mm, if_error_loc, n_list, write_csv, mm_factor, loc_factor, prob)
    results_dict["Error case"]["Scenario 3"]["top3_acc"] = item[0][1]
    results_dict["Error case"]["Scenario 3"]["ave_overhead"] = item[-1]
    results_dict["Error case"]["Scenario 3"]["ave_power"] = item[-2]

    mm_model_list = [mm_model_root_0]
    loc_model_list = [loc_model_root]
    detect_model_list = [detect_model_root]
    if_mm = 1
    if_loc = 1
    if_error_mm = 0
    if_error_loc = 1
    write_csv = 0
    n_list = [1, 3, 5, 6, 7]
    item = demo_class(scenario, params_list, mm_model_list, loc_model_list, detect_model_list,
                       if_mm,
                       if_loc,
                       if_error_mm, if_error_loc, n_list, write_csv, mm_factor, loc_factor, prob)
    results_dict["Correct case"]["Scenario 3"]["top3_acc"] = item[0][1]
    results_dict["Correct case"]["Scenario 3"]["ave_overhead"] = item[-1]
    results_dict["Correct case"]["Scenario 3"]["ave_power"] = item[-2]

    # Model on S4
    scenario = r'Scenario4'
    mm_model_root_0 = r'..\\..\\Mmwave_aided_model\\Model_beam_64\\Model_on_S4\\mm_GRU_BP_20.joblib'
    loc_model_root = r'..\\..\\Location_aid_model\\Model_beam_64\\Model_on_S4\\loc_RFF_BM_10.joblib'
    detect_model_root = r'..\\..\\Candidate_detect\\Model_on_S4\\User_detect_BP_110.joblib'

    params_list = {"N_r": N_r,
                   "b_th": 5,
                   "notx_l_th": 0.5,
                   "k": 8,
                   "m": 15,
                   "Scenario": "Scenario 4"}
    mm_model_list = [mm_model_root_0]
    loc_model_list = [loc_model_root]
    detect_model_list = [detect_model_root]
    if_mm = 0
    if_loc = 1
    if_error_mm = 0
    if_error_loc = 1
    write_csv = 0
    n_list = [1, 3, 5, 6, 7]

    item = demo_class(scenario, params_list, mm_model_list, loc_model_list, detect_model_list,
                       if_mm,
                       if_loc,
                       if_error_mm, if_error_loc, n_list, write_csv, mm_factor, loc_factor, prob)
    results_dict["Error case"]["Scenario 4"]["top3_acc"] = item[0][1]
    results_dict["Error case"]["Scenario 4"]["ave_overhead"] = item[-1]
    results_dict["Error case"]["Scenario 4"]["ave_power"] = item[-2]

    mm_model_list = [mm_model_root_0]
    loc_model_list = [loc_model_root]
    detect_model_list = [detect_model_root]
    if_mm = 1
    if_loc = 1
    if_error_mm = 0
    if_error_loc = 1
    write_csv = 0
    n_list = [1, 3, 5, 6, 7]

    item = demo_class(scenario, params_list, mm_model_list, loc_model_list, detect_model_list,
                       if_mm,
                       if_loc,
                       if_error_mm, if_error_loc, n_list, write_csv, mm_factor, loc_factor, prob)
    results_dict["Correct case"]["Scenario 4"]["top3_acc"] = item[0][1]
    results_dict["Correct case"]["Scenario 4"]["ave_overhead"] = item[-1]
    results_dict["Correct case"]["Scenario 4"]["ave_power"] = item[-2]

    # Model on S5
    scenario = r'Scenario5'
    mm_model_root_0 = r'..\\..\\Mmwave_aided_model\\Model_beam_64\\Model_on_S5\\mm_GRU_BP_27.joblib'
    loc_model_root = r'..\\..\\Location_aid_model\\Model_beam_64\\Model_on_S5\\loc_RFF_BM_11.joblib'
    detect_model_root = r'..\\..\\Candidate_detect\\Model_on_S5\\User_detect_BP_11.joblib'

    params_list = {"N_r": N_r,
                   "b_th": 5,
                   "notx_l_th": 0.5,
                   "k": 6,
                   "m": 9,
                   "Scenario": "Scenario 5"}
    mm_model_list = [mm_model_root_0]
    loc_model_list = [loc_model_root]
    detect_model_list = [detect_model_root]
    if_mm = 0
    if_loc = 1
    if_error_mm = 0
    if_error_loc = 1
    write_csv = 0
    n_list = [1, 3, 5, 6, 7]
    item = demo_class(scenario, params_list, mm_model_list, loc_model_list, detect_model_list,
                       if_mm,
                       if_loc,
                       if_error_mm, if_error_loc, n_list, write_csv, mm_factor, loc_factor, prob)
    results_dict["Error case"]["Scenario 5"]["top3_acc"] = item[0][1]
    results_dict["Error case"]["Scenario 5"]["ave_overhead"] = item[-1]
    results_dict["Error case"]["Scenario 5"]["ave_power"] = item[-2]

    mm_model_list = [mm_model_root_0]
    loc_model_list = [loc_model_root]
    detect_model_list = [detect_model_root]
    if_mm = 1
    if_loc = 1
    if_error_mm = 0
    if_error_loc = 1
    write_csv = 0
    n_list = [1, 3, 5, 6, 7]
    item = demo_class(scenario, params_list, mm_model_list, loc_model_list, detect_model_list,
                       if_mm,
                       if_loc,
                       if_error_mm, if_error_loc, n_list, write_csv, mm_factor, loc_factor, prob)
    results_dict["Correct case"]["Scenario 5"]["top3_acc"] = item[0][1]
    results_dict["Correct case"]["Scenario 5"]["ave_overhead"] = item[-1]
    results_dict["Correct case"]["Scenario 5"]["ave_power"] = item[-2]

    return results_dict



def error_mm_test_sim():
    if not os.path.exists('Error_correction_comparison_mm'):
        os.makedirs('Error_correction_comparison_mm')
    prob_list = np.arange(0, 6) / 5
    N_r_list = np.arange(0, 7, 1)
    for idxi, N_r in enumerate(N_r_list):
        overhead_matrix = np.zeros([prob_list.shape[0], 4 * 2])
        power_matrix = np.zeros([prob_list.shape[0], 4 * 2])
        for idxj, prob in enumerate(prob_list):
            # if idxj != 0:
            #     continue
            last_dict = error_mm_test(N_r, prob)
            for idxc, case in enumerate(["Error case", "Correct case"]):
                for idxk, scenario in enumerate(["Scenario 3", "Scenario 4", "Scenario 5", "Scenario 8"]):
                    overhead_matrix[idxj, idxk + idxc * 4] = last_dict[case][scenario]["ave_overhead"]
                    power_matrix[idxj, idxk + idxc * 4] = last_dict[case][scenario]["ave_power"]

        all_data = np.concatenate([overhead_matrix, power_matrix], axis=-1)
        df = pd.DataFrame(all_data, index=list(prob_list),
                          columns=["S3 overhead error", "S4 overhead error", "S5 overhead error", "S8 overhead error",
                                   "S3 overhead correct", "S4 overhead correct", "S5 overhead correct",
                                   "S8 overhead correct",
                                   "S3 power error", "S4 power error", "S5 power error", "S8 power error",
                                   "S3 power correct", "S4 power correct", "S5 power correct", "S8 power correct"
                                   ])
        df.to_csv('Error_correction_comparison_mm//Error_correction_comparison_mm_N_r_' + str(N_r) + '.csv',)





def error_loc_test_sim():
    if not os.path.exists('Error_correction_comparison_loc'):
        os.makedirs('Error_correction_comparison_loc')
    prob_list = np.arange(0, 6) / 5
    N_r_list = np.arange(0, 7, 1)
    for idxi, N_r in enumerate(N_r_list):
        overhead_matrix = np.zeros([prob_list.shape[0], 4 * 2])
        power_matrix = np.zeros([prob_list.shape[0], 4 * 2])
        for idxj, prob in enumerate(prob_list):
            last_dict = error_loc_test(N_r, prob)
            for idxc, case in enumerate(["Error case", "Correct case"]):
                for idxk, scenario in enumerate(["Scenario 3", "Scenario 4", "Scenario 5", "Scenario 8"]):
                    overhead_matrix[idxj, idxk + idxc * 4] = last_dict[case][scenario]["ave_overhead"]
                    power_matrix[idxj, idxk + idxc * 4] = last_dict[case][scenario]["ave_power"]

        all_data = np.concatenate([overhead_matrix, power_matrix], axis=-1)
        df = pd.DataFrame(all_data, index=list(prob_list),
                          columns=["S3 overhead error", "S4 overhead error", "S5 overhead error", "S8 overhead error",
                                   "S3 overhead correct", "S4 overhead correct", "S5 overhead correct",
                                   "S8 overhead correct",
                                   "S3 power error", "S4 power error", "S5 power error", "S8 power error",
                                   "S3 power correct", "S4 power correct", "S5 power correct", "S8 power correct"
                                   ])
        df.to_csv('Error_correction_comparison_loc//Error_correction_comparison_loc_N_r_' + str(N_r) + '.csv',)


if __name__ == "__main__":
    mm_factor = 10 * np.log10(1.2)
    loc_factor = 0.15
    error_mm_test_sim()
    error_loc_test_sim()

