import cv2
import albumentations as A


class BaseAugmentation(object):
    def __init__(self, img_size, is_train):
        self.is_train = is_train
        self.img_size = img_size
        self.transforms = self.get_transforms()

    def __call__(self, image):
        inputs = {"image": image}

        if self.transforms is not None:
            result = self.transforms(**inputs)
            image = result["image"]

        return image

    def get_transforms(self):
        if self.is_train:
            return A.Compose(
                [
                    A.Resize(self.img_size[0], self.img_size[1]),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(self.img_size[0], self.img_size[1]),
                ]
            )


class TrainAugmentation(BaseAugmentation):
    def get_transforms(self):
        if self.is_train:
            return A.Compose(
                [
                    A.Resize(self.img_size[0], self.img_size[1]),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, p=0.7),
                    A.OneOf([A.Blur(blur_limit=3, p=1.0), A.MedianBlur(blur_limit=3, p=1.0)], p=0.2),
                    A.CLAHE(clip_limit=(1, 4), p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.3),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(self.img_size[0], self.img_size[1]),
                ]
            )



class HardAugmentation(BaseAugmentation):
    def get_transforms(self):
        if self.is_train:
            return A.Compose(
                [
                    A.HorizontalFlip(p=0.5),  
                    A.VerticalFlip(p=0.2),  
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=60, border_mode=cv2.BORDER_REFLECT, p=0.8),  
                    A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.5),

                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.7),  
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5),  
                    A.ChannelShuffle(p=0.1),

                    A.MotionBlur(blur_limit=3, p=0.2),
                    A.Resize(self.img_size[0], self.img_size[1]),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(self.img_size[0], self.img_size[1]),
                ]
            )