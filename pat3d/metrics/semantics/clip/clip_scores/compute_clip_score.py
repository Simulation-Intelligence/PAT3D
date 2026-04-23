import json 
import numpy as np
import os 


if __name__ == "__main__":

    baseline_methods = ['blenderMCP', 'graphdreamer', 'midi', 'pat3d']

    folder = '/media/guyinglin/yihao/guying_sim_gen/mask_loss/rebuttal/clip_scores'

    for method in baseline_methods:
        method_list = []
        method_json_file = os.path.join(folder, f'clip_score_{method}.json')
        with open(method_json_file, 'r') as f:
            method_data = json.load(f)
            method_data = method_data[method]
            for case_name in method_data.keys():
                #print(case_name)
                case_score_list = method_data[case_name]['list']
                method_list.append(case_score_list)
        
        method_list = np.array(method_list)
        ## print the min, max, median of the method_list
        print(f"{method} min: {np.min(method_list)}")
        print(f"{method} max: {np.max(method_list)}")
        print(f"{method} median: {np.median(method_list)}")