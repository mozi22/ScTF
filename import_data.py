import numpy as np

def prepare_input_data(img1, img2, data_format):
    """Creates the arrays used as input from the two images."""
    # scale images if necessary
    if img1.size[0] != 256 or img1.size[1] != 192:
        img1 = img1.resize((256,192))
    if img2.size[0] != 256 or img2.size[1] != 192:
        img2 = img2.resize((256,192))
    
    if data_format == 'channels_first':
        img1_arr = img1_arr.transpose([2,0,1])
        img2_arr = img2_arr.transpose([2,0,1])
        img2_2_arr = img2_2_arr.transpose([2,0,1])
        image_pair = np.concatenate((img1_arr,img2_arr), axis=0)
    else:
        image_pair = np.concatenate((img1_arr,img2_arr),axis=-1)
    
    result = {
        'image_pair': image_pair[np.newaxis,:],
        'image1': img1_arr[np.newaxis,:], # first image
        'image2_2': img2_2_arr[np.newaxis,:], # second image with (w=64,h=48)
    }
    return result



def read_flo_file(file_path):
	with open(file_path, 'rb') as f:

		magic = np.fromfile(f, np.float32, count=1)

		if 202021.25 != magic:
			print('Magic number incorrect. Invalid .flo file')
		else:
			w = np.fromfile(f, np.int32, count=1)[0]
			h = np.fromfile(f, np.int32, count=1)[0]

			data = np.fromfile(f, np.float32, count=2*w*h)

			# Reshape data into 3D array (columns, rows, bands)
			data2D = np.resize(data, (w, h, 2))
			return data2D
