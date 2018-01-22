
import flow_test as ft

folder = '/home/muazzam/mywork/python/thesis/server/dataset_synthetic_sm/driving/'
# folder = '/misc/lmbraid19/muazzama/dataset_synthetic_sm/driving/'

img1 = folder + 'frames_finalpass_webp/35mm_focallength/scene_backwards/fast/left/0001.webp'
img2 = folder + 'frames_finalpass_webp/35mm_focallength/scene_backwards/fast/left/0002.webp'
depth1 = folder + 'disparity/35mm_focallength/scene_backwards/fast/left/0001.pfm'
depth2 = folder + 'disparity/35mm_focallength/scene_backwards/fast/left/0002.pfm'

predictor = ft.FlowPredictor()
predictor.preprocess(img1,img2,depth1,depth2)
predictor.predict()