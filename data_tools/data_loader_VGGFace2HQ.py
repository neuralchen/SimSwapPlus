import os
import torch
import random
from PIL import Image
from pathlib import Path
from torch.utils import data
from torchvision import transforms as T
# from StyleResize import StyleResize

class data_prefetcher():
    def __init__(self, loader):
        self.loader = loader
        self.dataiter = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.num_images = len(loader)
        self.preload()

    def preload(self):
        try:
            self.content = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            self.content = next(self.dataiter)
            
        with torch.cuda.stream(self.stream):
            self.content= self.content.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            # self.next_input = self.next_input.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        content = self.content
        self.preload()
        return content
    
    def __len__(self):
        """Return the number of images."""
        return self.num_images

class VGGFace2HQDataset(data.Dataset):
    """Dataset class for the Artworks dataset and content dataset."""

    def __init__(self,
                    content_image_dir,
                    selectedContent,
                    content_transform,
                    subffix='jpg',
                    random_seed=1234):
        """Initialize and preprocess the CelebA dataset."""
        self.content_image_dir  = content_image_dir
        self.content_transform  = content_transform
        self.selectedContent    = selectedContent
        self.subffix            = subffix
        self.content_dataset    = []
        self.random_seed        = random_seed
        self.preprocess()
        self.num_images = len(self.content_dataset)

    def preprocess(self):
        """Preprocess the Artworks dataset."""
        print("processing content images...")
        for dir_item in self.selectedContent:
            join_path = Path(self.content_image_dir,dir_item)
            if join_path.exists():
                print("processing %s"%dir_item,end='\r')
                images = join_path.glob('*.%s'%(self.subffix))
                for item in images:
                    self.content_dataset.append(item)
            else:
                print("%s dir does not exist!"%dir_item,end='\r')
        random.seed(self.random_seed)
        random.shuffle(self.content_dataset)
        print('Finished preprocessing the Content dataset, total image number: %d...'%len(self.content_dataset))

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        filename        = self.content_dataset[index]
        image           = Image.open(filename)
        content         = self.content_transform(image)
        return content

    def __len__(self):
        """Return the number of images."""
        return self.num_images

def GetLoader(  dataset_roots,
                batch_size=16,
                crop_size=512,
                **kwargs
                ):
    """Build and return a data loader."""
    if not kwargs:
        a = "Input params error!"
        raise ValueError(print(a))
        
    colorJitterEnable = kwargs["color_jitter"]
    colorConfig       = kwargs["color_config"]
    num_workers       = kwargs["dataloader_workers"]
    num_workers       = kwargs["dataloader_workers"]
    place365_root     = dataset_roots["Place365_big"]
    selected_c_dir    = kwargs["selected_content_dir"]
    random_seed       = kwargs["random_seed"]
    
    c_transforms = []
    
    # s_transforms.append(T.Resize(900))
    c_transforms.append(T.Resize(900))
    c_transforms.append(T.RandomCrop(crop_size))
    c_transforms.append(T.RandomHorizontalFlip())
    c_transforms.append(T.RandomVerticalFlip())

    if colorJitterEnable:
        if colorConfig is not None:
            print("Enable color jitter!")
            colorBrightness = colorConfig["brightness"]
            colorContrast   = colorConfig["contrast"]
            colorSaturation = colorConfig["saturation"]
            colorHue        = (-colorConfig["hue"],colorConfig["hue"])
            c_transforms.append(T.ColorJitter(brightness=colorBrightness,\
                                contrast=colorContrast,saturation=colorSaturation, hue=colorHue))
    c_transforms.append(T.ToTensor())
    c_transforms.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    c_transforms = T.Compose(c_transforms)

    content_dataset = Place365Dataset(
                            place365_root, 
                            selected_c_dir,
                            c_transforms,
                            "jpg",
                            random_seed)
    content_data_loader = data.DataLoader(dataset=content_dataset,batch_size=batch_size,
                    drop_last=True,shuffle=True,num_workers=num_workers,pin_memory=True)
    prefetcher = data_prefetcher(content_data_loader)
    return prefetcher

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

if __name__ == "__main__":
    from torchvision.utils import save_image
    style_class  = ["vangogh","picasso","samuel"]
    categories_names = \
        ['a/abbey', 'a/arch', 'a/amphitheater', 'a/aqueduct', 'a/arena/rodeo', 'a/athletic_field/outdoor',
         'b/badlands', 'b/balcony/exterior', 'b/bamboo_forest', 'b/barn', 'b/barndoor', 'b/baseball_field',
         'b/basilica', 'b/bayou', 'b/beach', 'b/beach_house', 'b/beer_garden', 'b/boardwalk', 'b/boathouse',
         'b/botanical_garden', 'b/bullring', 'b/butte', 'c/cabin/outdoor', 'c/campsite', 'c/campus',
         'c/canal/natural', 'c/canal/urban', 'c/canyon', 'c/castle', 'c/church/outdoor', 'c/chalet',
         'c/cliff', 'c/coast', 'c/corn_field', 'c/corral', 'c/cottage', 'c/courtyard', 'c/crevasse',
         'd/dam', 'd/desert/vegetation', 'd/desert_road', 'd/doorway/outdoor', 'f/farm', 'f/fairway',
         'f/field/cultivated', 'f/field/wild', 'f/field_road', 'f/fishpond', 'f/florist_shop/indoor',
         'f/forest/broadleaf', 'f/forest_path', 'f/forest_road', 'f/formal_garden', 'g/gazebo/exterior',
         'g/glacier', 'g/golf_course', 'g/greenhouse/indoor', 'g/greenhouse/outdoor', 'g/grotto', 'g/gorge',
         'h/hayfield', 'h/herb_garden', 'h/hot_spring', 'h/house', 'h/hunting_lodge/outdoor', 'i/ice_floe',
         'i/ice_shelf', 'i/iceberg', 'i/inn/outdoor', 'i/islet', 'j/japanese_garden', 'k/kasbah',
         'k/kennel/outdoor', 'l/lagoon', 'l/lake/natural', 'l/lawn', 'l/library/outdoor', 'l/lighthouse',
         'm/mansion', 'm/marsh', 'm/mausoleum', 'm/moat/water', 'm/mosque/outdoor', 'm/mountain',
         'm/mountain_path', 'm/mountain_snowy', 'o/oast_house', 'o/ocean', 'o/orchard', 'p/park',
         'p/pasture', 'p/pavilion', 'p/picnic_area', 'p/pier', 'p/pond', 'r/raft', 'r/railroad_track',
         'r/rainforest', 'r/rice_paddy', 'r/river', 'r/rock_arch', 'r/roof_garden', 'r/rope_bridge',
         'r/ruin', 's/schoolhouse', 's/sky', 's/snowfield', 's/swamp', 's/swimming_hole',
         's/synagogue/outdoor', 't/temple/asia', 't/topiary_garden', 't/tree_farm', 't/tree_house',
         'u/underwater/ocean_deep', 'u/utility_room', 'v/valley', 'v/vegetable_garden', 'v/viaduct',
         'v/village', 'v/vineyard', 'v/volcano', 'w/waterfall', 'w/watering_hole', 'w/wave',
         'w/wheat_field', 'z/zen_garden', 'a/alcove', 'a/apartment-building/outdoor', 'a/artists_loft',
         'b/building_facade', 'c/cemetery']

    s_datapath      = "D:\\F_Disk\\data_set\\Art_Data\\data_art_backup"
    c_datapath      = "D:\\Downloads\\data_large"
    savepath        = "D:\\PatchFace\\PleaseWork\\multi-style-gan\\StyleTransfer\\dataloader_test"
    
    imsize          = 512
    s_datasetloader= getLoader(s_datapath,c_datapath, 
                style_class, categories_names,
                crop_size=imsize, batch_size=16, num_workers=4)
    wocao           = iter(s_datasetloader)
    for i in range(500):
        print("new batch")
        s_image,c_image,label     = next(wocao)
        print(label)
        # print(label)
        # saved_image1 = torch.cat([denorm(image.data),denorm(hahh.data)],3)
        # save_image(denorm(image), "%s\\%d-label-%d.jpg"%(savepath,i), nrow=1, padding=1)
    pass
    # import cv2
    # import os
    # for dir_item in categories_names:
    #     join_path = Path(contentdatapath,dir_item)
    #     if join_path.exists():
    #         print("processing %s"%dir_item,end='\r')
    #         images = join_path.glob('*.%s'%("jpg"))
    #         for item in images:
    #             temp_path = str(item)
    #             # temp = cv2.imread(temp_path)
    #             temp = Image.open(temp_path)
    #             if temp.layers<3:
    #                 print("remove broken image...")
    #                 print("image name:%s"%temp_path)
    #                 del temp
    #                 os.remove(item)