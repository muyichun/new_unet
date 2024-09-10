from PIL import Image

from UNet1.architect.utils import keep_image_size_open


a = Image.open("/Users/muyichun/Desktop/3_210/0.png")
b = keep_image_size_open("/Users/muyichun/Desktop/3_210/0.png")

# b.save("/Users/muyichun/Desktop/3_210/111.png")
print(a)
print(b)

# mask.save("/Users/muyichun/Desktop/3_210/222.png")