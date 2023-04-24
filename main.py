import networkx as nx
import numpy as np
from LPSI import LPSI
from numpy import array
import matplotlib.pyplot as plt
from PIL import Image
import os


# 相关配置
G = nx.Graph()
G.add_nodes_from(range(16))
G.add_edges_from([(0, 1), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10),
                  (1,2), (1,3), (1,4),
                  (6,12), (7,13), (11, 12), (11, 13), (11, 14),
                  (12, 15),])
net_state = np.array([1,1,1,1,1,-1,-1,-1,-1,-1,-1,1,1,1,1,-1], dtype=np.float64)
alpha = 0.5
duration = 600 # in milliseconds
gif_file_name = f"LPSI_alpha_{alpha}.gif"
steps = 10
base_pos = {0: array([-0.16269107,  0.11653528]), 1: array([0.04049017, 0.44984379]), 2: array([0.37858381, 0.51762434]), 3: array([0.17964717, 0.75448336]), 4: array([-0.13831026,  0.77087993]), 5: array([-0.27603728,  0.37786808]), 6: array([ 0.14897254, -0.06600463]), 7: array([-0.16898718, -0.2267173 ]),
       8: array([-0.48160387,  0.29493655]), 9: array([-0.53379611,  0.08559912]), 10: array([-0.43551389, -0.09201309]), 11: array([ 0.21834619, -0.68199002]), 12: array([ 0.35578068, -0.35889629]), 13: array([-0.09580418, -0.55766496]), 14: array([ 0.28105419, -1.        ]), 15: array([ 0.6898691 , -0.38448419])}
node_pos_offset = {8: array([0, 0.2]), 10: array([0, -0.2])}
pos = {i: base_pos[i] + node_pos_offset.get(i, array([0, 0])) for i in base_pos}
label_pos_offset_base = array([0.1, 0.1])
label_pos_offset = {0: array([0, -0.06]),
                    1: array([0, -0.2]),
                    15: array([-0.1, 0.1]),
                    4: array([0, -0.1]),
                    8: array([0, -0.05]),
                    10: array([0, -0.1]),
                    11: array([0, -0.1])}
label_pos = {i: base_pos[i] + label_pos_offset_base + node_pos_offset.get(i, array([0, 0])) + label_pos_offset.get(i, array([0, 0])) for i in base_pos}

# 开始生成图像
adjacent_Matrix = nx.to_numpy_matrix(G)
node_color = list()
for i in net_state:
    if i < 0:
        node_color.append('g')
    else:
        node_color.append('r')

if pos is None:
    pos = nx.kamada_kawai_layout(G)# nx.spring_layout(G)#, seed=1)
print("POS", pos)

def draw(r, note=''):
    node_labels = dict()
    node_size = list()#dict()
    for index, r_i in enumerate(r):
        node_labels[index] = f'{r_i:.2f}'
        # node_size[index] = int(500 + 300 * abs(r_i))
        # node_size.append(float(500 + 500 * r_i))
        node_size.append(float(50 + 800 * abs(r_i)))
    
    plt.figure(figsize=(8, 4), dpi=300)

    nx.draw_networkx(G, pos, arrows=True,
                     with_labels=True, #False,  # True的话会显示节点ID或者下面的labels
                     nodelist=list(G.nodes()),  # 基本参数
                     node_color=node_color, node_size=node_size, alpha=1,  # 结点参数,alpha是透明度
                     width=1, style='solid',
                     # 边参数(solid|dashed|dotted,dashdot)
                    #  labels=node_labels,
                     font_size=10, font_weight='normal',
                    #  label=['state']
                     )
    # 画出label
    nx.draw_networkx_labels(G, label_pos, labels=node_labels, font_size=10, font_weight='normal')
    plt.title(f"LPSI Step {note:02d}" if note > 0 else "LPSI Start")
    # plt.show()
    plt.savefig(f"LPSI_{note:02d}_alpha_{alpha}.png")
    
draw(net_state, 0)
lpsi = LPSI(adjacent_Matrix, alpha, net_state)
for i, r in enumerate(lpsi.step_converge()):
    print("Step", i + 1, ":", np.array(r))
    draw(r, i+1)
    if i + 1 >= steps:
        break


# 生成GIF
filenames = [f'LPSI_{i:02d}_alpha_{alpha}.png' for i in range(steps + 1)]
filenames = [filenames[0]] * 5 + filenames + [filenames[-1]] * 5  # 最开始和最后暂停一下
images = []
for filename in filenames:
    images.append(Image.open(filename))

images[0].save(gif_file_name,
               save_all=True, 
               append_images=images[1:],
               duration=duration,
               loop=0)

# 清理png文件
for f in set(filenames):
    os.remove(f)
