from PIL import Image


def my_image_open(path, size=(256, 256)):
    return Image.open(path)

def keep_image_size_open(path, size=(256, 256)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('L', (temp, temp))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask


def keep_image_size_open_rgb(path, size=(256, 256)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('RGB', (temp, temp), (0, 0, 0))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask


'''
    仅测试
'''
# if __name__ == '__main__':
#     img = keep_image_size_open_rgb('train_image/0.png')
#     img.show()