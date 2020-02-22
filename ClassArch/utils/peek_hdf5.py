import h5py


file_name = '../data/modelnet40/ply_data_labeled0.h5'

final_label = h5py.File(file_name, 'r')
print(final_label)
print(final_label['data'])
print(final_label['label'])
