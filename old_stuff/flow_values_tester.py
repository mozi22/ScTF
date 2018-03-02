import numpy as np
from PIL import Image


# orig_u = Image.open('originalflow_u.tiff')
# orig_v = Image.open('originalflow_v.tiff')

# input_u = Image.open('flow_u_2.tiff')
# input_v = Image.open('flow_v2.tiff')
# input_u = Image.open('test_flowu.tiff')
# input_v = Image.open('test_flowv.tiff')

norm = Image.open('opf_1.tiff')
denorm = Image.open('opf_2.tiff')

inp = np.array(norm)
out = np.array(denorm)

out = np.delete(out,(223,222,221,220),axis=0)
out = np.delete(out,(0,1,2,3),axis=0)



for i in range(0,out.shape[0]):
	for j in range(0,inp.shape[0]):
		if not out[i,j]==inp[i,j]:
			print('inp = ' + str(inp[i,j]))
			print('out = ' + str(out[i,j]))


# print(ou)
# print(iu)
# # print(iu.shape)

print(np.array_equal(out,inp))