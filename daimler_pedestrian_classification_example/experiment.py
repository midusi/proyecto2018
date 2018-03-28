import sys


import skimage.feature
import numpy as np
import collections
BoundingBox = collections.namedtuple("BoundingBox",["r","c","h","w"])




def my_hog(image,visualise=False):    
    return skimage.feature.hog(image,block_norm='L2-Hys',transform_sqrt=True,visualise=visualise)
 
   
    
def calculate_descriptor_windows(image,descriptor_function,window_scales=[(90,90)],window_strides=(40,40)):
    
    rows=image.shape[0]//window_strides[0]
    cols=image.shape[1]//window_strides[1]
    descriptors=[]
    for i in range(rows):
        ai=i*window_strides[0]
        for j in range(cols):
            aj=j*window_strides[1]
            for scale in window_scales:
                to=(ai+scale[0],aj+scale[1])
                if (to[0]<image.shape[0] and to[1]<image.shape[1]):
                    window=image[ai:to[0],aj:to[1]]
                    descriptor=descriptor_function(window)
                    bb=BoundingBox(ai,aj,scale[0],scale[1])
                    descriptors.append((bb,descriptor))
    
    return descriptors

def hogs_uniform(image,crop_grid_size,crop_size):
    n=len(images)
    ch,cw=crop_size
    gh,gw=crop_grid_size
    crops_per_image=gh*gw
    bboxes=[]
    sample_hog=my_hog(image[0:ch,0:cw])
    
    hogs=np.zeros((gh*gw,sample_hog.shape[0]))
    j=0
    h,w=image.shape
    assert ch+gh<=h and cw+gw<=w
    adjusted_h=h-ch
    adjusted_w=w-cw
    h_points=np.linspace(0,adjusted_h,gh, dtype = int, endpoint=False)
    w_points=np.linspace(0,adjusted_w,gw, dtype = int, endpoint=False)
    assert len(h_points) == gh
    assert len(w_points) == gw
    for t in h_points:
        for l in w_points:
            bboxes.append((t,l,ch,cw))
            crops[j,:,:]=image[t:t+ch,l:l+cw]
            j+=1
    return hogs,bboxes

def hogs_from_images(images):
            
    first_image=images[0]
    print("image shape %s" % str(first_image.shape))

    hog=my_hog(first_image)
    print("resulting hog size = %d" % len(hog))
    n=len(images)
    hogs=np.zeros((n,hog.shape[0]))

    print("Generating hogs for %d images..." % n)
    for i in range(n):
        hogs[i,:]=my_hog(images[i])
        if i % (n//10)==0:
            print("  %f .." % (i/n*100))
    print("Done")    
    return hogs


def crop_images_uniform(images,crop_grid_size,crop_size):
    n=len(images)
    ch,cw=crop_size
    gh,gw=crop_grid_size
    crops_per_image=gh*gw
    crops=np.zeros((n*crops_per_image,ch,cw))
    j=0
    for i in range(n):
        image=images[i]
        h,w=image.shape
        assert ch+gh<=h and cw+gw<=w
        adjusted_h=h-ch
        adjusted_w=w-cw
        h_points=np.linspace(0,adjusted_h,gh, dtype = int, endpoint=False)
        w_points=np.linspace(0,adjusted_w,gw, dtype = int, endpoint=False)
        assert len(h_points) == gh
        assert len(w_points) == gw
        for t in h_points:
            for l in w_points:
                crops[j,:,:]=image[t:t+ch,l:l+cw]
                j+=1
    return crops
    
