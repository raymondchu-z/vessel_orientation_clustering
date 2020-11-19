import time
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
# import pixellib
from pixellib.semantic import semantic_segmentation
start = time.time()
img_path = "829912.jpg"
segment_image = semantic_segmentation()
segment_image.load_pascalvoc_model("model/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
segment_image.segmentAsPascalvoc(img_path, output_image_name = "image_new.jpg",overlay = True)

end = time.time()
print(f"Inference Time: {end-start:.2f}seconds") 