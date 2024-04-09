# ACDWM (Adaptive Chunk-based Dynamic Weighted Majority)
# Example: python main.py sea_abrupt.npz

from numpy import *
from chunk_size_select import *
from clf_ACDWM import ACDWM
from chunk_based_methods import *
from check_measure import *
import matplotlib.pyplot as plt
import sys

# data_name = sys.argv[1]
data_name = "sea_abrupt.npz"
load_data = load('data/' + data_name)
data = load_data['data']
label = load_data['label']
reset_pos = load_data['reset_pos'].astype(int)

data_num = data.shape[0]
chunk_size = 1000

run_num = 1

pq_result_acdwm = [{} for _ in range(run_num)]

for run_i in range(run_num):

    acss = ChunkSizeSelect()
    model_acdwm = ACDWM(data_num=data_num, chunk_size=0)
    pred_acdwm = array([])

    print('Round ' + str(run_i))
    for i in range(data_num):
        print(i)
        acss.update(data[i], label[i])
        if i == data_num - 1:
            chunk_data, chunk_label = acss.get_chunk_2()
            pred_acdwm = append(pred_acdwm, model_acdwm.predict(chunk_data))
        elif acss.get_enough() == 1:
            chunk_data, chunk_label = acss.get_chunk()
            pred_acdwm = append(pred_acdwm, model_acdwm.update_chunk(chunk_data, chunk_label))

    pq_result_acdwm[run_i] = prequential_measure(pred_acdwm, label, reset_pos)

print('acdwm: %f' % mean([pq_result_acdwm[i]['gm'][-1] for i in range(run_num)]))
print(pq_result_acdwm)