
import flow_test as ft
import helpers as hpl
import numpy as np
from   PIL import Image

folder = '/home/muazzam/mywork/python/thesis/server/dataset_synthetic/driving/'
# folder = '/misc/lmbraid19/muazzama/dataset_synthetic_sm/driving/'

img1 = folder + 'frames_finalpass_webp/35mm_focallength/scene_backwards/fast/left/0001.webp'
img2 = folder + 'frames_finalpass_webp/35mm_focallength/scene_backwards/fast/left/0002.webp'
disparity1 = folder + 'disparity/35mm_focallength/scene_backwards/fast/left/0001.pfm'
disparity2 = folder + 'disparity/35mm_focallength/scene_backwards/fast/left/0002.pfm'

predictor = ft.FlowPredictor()
predictor.preprocess(img1,img2,disparity1,disparity2)
predictor.predict()



# for testing with ground truth

opt_flow = folder + 'optical_flow/35mm_focallength/scene_backwards/fast/into_future/left/OpticalFlowIntoFuture_0004_L.pfm'
disp_change = folder + 'disparity_change/35mm_focallength/scene_backwards/fast/into_future/left/0001.pfm'




# lbl = predictor.read_gt(opt_flow,disp_change)
# opt_flow = np.pad(lbl,((4,4),(0,0),(0,0)),'constant')
# predictor.postprocess(flow=opt_flow,show_flow=True,gt=True)





# for testing

# opt = hpl.readPFM(opt_flow)[0]
# Image.fromarray(opt.astype(np.uint8)).show()






# dispar1 = hpl.readPFM(disparity1)[0]
# dispar2 = hpl.readPFM(disparity2)[0]
opt_flow = hpl.readPFM(opt_flow)[0]
# dispar_chng = hpl.readPFM(disp_change)[0]
# result1 = predictor.get_depth_from_disp(dispar1)
# result2 = predictor.get_depth_from_disp(dispar2)
# result3 = predictor.get_depth_from_disp(dispar_chng)
# result3 = predictor.get_depth_chng_from_disp_chng(dispar1,dispar_chng)
# Image.open(img1).show()
# Image.open(img2).show()
# Image.fromarray(result1).show()
# Image.fromarray(result2).show()
# Image.fromarray(result3).show()
# Image.fromarray(opt_flow[:,:,0]).show()
# Image.fromarray(opt_flow[:,:,1]).show()
