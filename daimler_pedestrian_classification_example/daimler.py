import collections 
import os
import skimage.io as io
import matplotlib.patches as patches
import matplotlib.pyplot as plt

ImageMetadata = collections.namedtuple('ImageMetadata', ['filename', 'objects'])
ObjectMetadata = collections.namedtuple('ObjectMetadata', ['id', 'type','confidence','bbox'])

def file_to_lines(filepath):
    with open(filepath) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content] 
    return content

def get_object_types():
    object_types={0:"fully-visible pedestrian",
                1:"bicyclist",
                2:"motorcyclist",
                10:"pedestrian group",
                255:"partially visible pedestrian, bicyclist, motorcyclist",}
    return object_types

#metadata_filepath: path for the metadata file
#Returns
#metadata is a list of ImageMetadata objects, that contain the filename and object information (bbox, confidence, etc) of a test image
#object_types is a dictionary that maps object type ids to their string representation (ie,  1:"bicyclist")

colors={0:"yellow",
                1:"blue",
                2:"cyan",
                10:"orange",
                255:"red",}

def draw_legends(ax,position=(5,5)):
    object_types=get_object_types()
    x,y=position
    w=15
    h=10
    padding=5
    for (k,color) in colors.items():
        object_type=object_types[k]
        ax.add_patch(patches.Rectangle((x,y),w,h,color=color  ))
        ax.text(x+w+5,y+h,object_type,color="#2efe07",fontsize=12)
        y=y+h+padding
            
def draw_bounding_boxes(ax,objects):
    object_types=get_object_types()
    for obj in objects:
        if obj.confidence >=0:#objects with confidence=0 can be ignored
            x,y,x1,y1=obj.bbox
            w=x1-x
            h=y1-y
            color=colors[obj.type]
            ax.add_patch(patches.Rectangle((x,y),w,h,fill=False,edgecolor=color  ))
            #object_type=object_types[obj.type]
            #ax.text(x,y,object_type,color=color)
    draw_legends(ax)
def display_image_with_bounding_boxes(image,image_metadata):
    fig = plt.figure(figsize=(14,10))
    ax = fig.add_subplot(111, aspect='equal')
    ax.imshow(image)
    draw_bounding_boxes(ax,image_metadata.objects)
    
def read_image_metadata(metadata_filepath):
    image_metadata=[]
    lines=file_to_lines(metadata_filepath)
    lines=lines[4:]
    i=0

    while i<len(lines):
        assert lines[i]==";"
        image_filename=lines[i+1]
        #image_resolution=lines[i+2]
        zero_str,object_count_str=lines[i+3].split(" ")
        assert zero_str=="0"
        object_count=int(object_count_str)
        objects=[]
        i=i+4
        for j in range(object_count):
            garbage,object_type_str=lines[i].split(" ")
            object_type=int(object_type_str)
            object_id_str,uniqueid=lines[i+1].split(" ")
            object_id=int(object_id_str)
            confidence=float(lines[i+2])
            bbox= [float(x) for x in list(lines[i+3].split(" "))]
            objects.append(ObjectMetadata(object_id,object_type,confidence,bbox))
            i+=5
        image_metadata.append(ImageMetadata(image_filename,objects))
        
    
    return image_metadata,get_object_types()

def read_image_directory(folderpath):
#     images=[]
    images = io.imread_collection(folderpath+'/*.pgm')
    return images
         
#basepath is the path of the extracted tar of the dataset
#Returns:
#train_pedestrian is a list of segmented images of pedestrians.
#train_non_pedestrian is a list of images where we are sure that there are NO pedestrians.
#metadata is a list of ImageMetadata objects, that contain the filename and object information (bbox, confidence, etc) of a test image
#object_types is a dictionary that maps object type ids to their string representation (ie,  1:"bicyclist")    
def get_dataset(basepath):
    metadata_filepath=os.path.join(basepath,"GroundTruth/GroundTruth2D.db")
    test_metadata,object_types=read_image_metadata(metadata_filepath)
    
    train_pedestrian_folderpath=os.path.join(basepath,"Data/TrainingData/Pedestrians/48x96/")
    train_non_pedestrian_folderpath=os.path.join(basepath,"Data/TrainingData/NonPedestrians/")
    test_folderpath=os.path.join(basepath,"Data/TestData")
    
    train_pedestrian=read_image_directory(train_pedestrian_folderpath)
    train_non_pedestrian=read_image_directory(train_non_pedestrian_folderpath)
    
    return train_pedestrian,train_non_pedestrian,test_metadata,object_types
    
    