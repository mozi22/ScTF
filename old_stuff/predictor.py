
import flow_test as ft
import helpers as hpl
import numpy as np
from   PIL import Image

folder = '/home/muazzam/mywork/python/thesis/server/dataset_synthetic/driving/'
# folder = '/misc/lmbraid19/muazzama/dataset_synthetic_sm/driving/'

img1 = folder + 'frames_finalpass_webp/35mm_focallength/scene_backwards/fast/left/0001.webp'
img2 = folder + 'frames_finalpass_webp/35mm_focallength/scene_backwards/fast/left/0002.webp'
depth1 = folder + 'disparity/35mm_focallength/scene_backwards/fast/left/0001.pfm'
depth2 = folder + 'disparity/35mm_focallength/scene_backwards/fast/left/0002.pfm'

predictor = ft.FlowPredictor()
predictor.preprocess(img1,img2,depth1,depth2)
# predictor.predict()



# for testing with ground truth

opt_flow = folder + 'optical_flow/35mm_focallength/scene_backwards/fast/into_future/left/OpticalFlowIntoFuture_0001_L.pfm'
disp_change = folder + 'disparity_change/35mm_focallength/scene_backwards/fast/into_future/left/0002.pfm'

predictor.print_flow(opt_flow)
# lbl = predictor.read_gt(opt_flow,disp_change)
# opt_flow = np.pad(lbl,((4,4),(0,0),(0,0)),'constant')
# predictor.postprocess(opt_flow)