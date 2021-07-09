import albumentations as A
from albumentations.pytorch import ToTensorV2

# albumentation transform
def albumentation_transform(train, mean, stddev):
    # Scaling mean for 255 range 
    mean_255 = [255*i for i in mean]
    if train:
        transform = A.Compose([ 
                           A.Normalize(mean=mean, std=stddev, max_pixel_value=255.0), 
                           A.PadIfNeeded(min_height=36, min_width=36, value=mean, always_apply=True),
                           A.RandomCrop(height=32, width=32, always_apply=True, p=1.0),
                           A.HorizontalFlip(always_apply=False, p=0.5),
                        #    A.ShiftScaleRotate (shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                           A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, fill_value=mean),
                           # A.Rotate(limit=5, always_apply=True, p=1.0),
                        #    A.ToGray(),
                           ToTensorV2(),
                           ])
    else:
        transform = A.Compose([                                                     
                           A.Normalize(mean=mean, std=stddev, max_pixel_value=255.0),
                           ToTensorV2(),
                           ])

    return transform