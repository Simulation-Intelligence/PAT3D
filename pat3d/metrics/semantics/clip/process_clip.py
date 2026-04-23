import os
import json
import numpy as np

pat3d_path = '/media/guyinglin/yihao/guying_sim_gen/mask_loss/rebuttal/clip_score_pat3d.json'
graphdreamer_path = '/media/guyinglin/yihao/guying_sim_gen/mask_loss/rebuttal/clip_score_graphdreamer.json'
midi_path = '/media/guyinglin/yihao/guying_sim_gen/mask_loss/rebuttal/clip_score_midi.json'
blenderMCP_path = '/media/guyinglin/yihao/guying_sim_gen/mask_loss/rebuttal/clip_score_blenderMCP.json'

if __name__ == "__main__":
    baseline_methods = ['pat3d', 'graphdreamer', 'midi', 'blenderMCP']
    for baseline_method in baseline_methods:
        all_scores = []
        score_data = json.load(open(f'/media/guyinglin/yihao/guying_sim_gen/mask_loss/rebuttal/clip_score_{baseline_method}.json', 'r'))
        for scene_case in score_data[baseline_method].keys():
            score_mean = score_data[baseline_method][scene_case]['mean']
            all_scores.append(score_mean)
        baseline_mean = np.mean(all_scores)
        print(f"{baseline_method}: {baseline_mean:.4f}")
        print('--------------------------------')



    