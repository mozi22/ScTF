import h5py
filename = 'sun3d_train_0.01m_to_0.1m.h5'
f = h5py.File(filename, 'r')

# List all groups
a_group_key = list(f.keys())[0]
print(a_group_key)
# Get the data
data = list(f[a_group_key])

print(data)