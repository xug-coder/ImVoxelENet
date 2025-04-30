import pickle

with open('/mmdetection3d/data/scannet/scannet_infos_val.pkl', 'rb') as fin:
    a = pickle.load(fin)
print(a[0])
print(len(a[0]))
print(a.keys())
print(a.keys())