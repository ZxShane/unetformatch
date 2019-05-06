# -*- coding:utf-8 -*-

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import glob
import SimpleITK as sitk
from matplotlib import pyplot as plt


def to_hu(image):
    MIN_BLOOD = -400
    MAX_BLOOD = 400
    image = (image - MIN_BLOOD) / (MAX_BLOOD - MIN_BLOOD)
    image[image > 1] = 1.
    image[image < 0.5] = 0.
    return image

print('-' * 30)
print('Creating training images...')
print('-' * 30)
cnt = 1000;
for j in range(66, 70):
    number = cnt + j
    filname = str(number)
    file = "train/" + filname + "/venous phase/200"
    imgs = glob.glob("train/" + filname + "/venous phase" + "//*." + "dcm")
    print(len(imgs))
    imgdatas = np.ndarray((len(imgs), 512, 512, 1), dtype=np.uint8)
    imglabr = np.ndarray((len(imgs), 512, 512, 1), dtype=np.uint8)
    i = 1
    for imgname in imgs:
        name = ""
        if i < 10:
            numname = "0" + str(i)
        else:
            numname = str(i)
        name = file + numname + ".dcm"
        # pngname = file + numname + "_mask.png"
        print(imgname)
        image = sitk.ReadImage(name)
        image_array = sitk.GetArrayFromImage(image)  # z, y, x
        print(image_array.shape)
        image_array = image_array.swapaxes(0, 2)
        image_array = image_array.swapaxes(0, 1)
        # 可以使用 transpose
        # 维数变化
        print(image_array.shape)

        image_array = to_hu(image_array)
        images = np.squeeze(image_array)

        plt.imshow(images,cmap="gray")
        plt.axis("off")
        plt.savefig(name+"_new.png")
        print("image_array done")
        plt.show()
        imgdatas[i - 1] = image_array

        print(imgname)
        # img = load_img(pngname, grayscale=True)
        img = img_to_array(img)
        # label = img_to_array(img)

        imglabr[i - 1] = label

        i += 1

        print('loading done')

        # np.save('train/' + filname + '/imgs_mask_train.npy', imglabr)
        print("imgs_mas_train_1001.npy done ")
        np.save('train/' + filname + '/imgs_train.npy', imgdatas)
        print('Saving to .npy files done.')
