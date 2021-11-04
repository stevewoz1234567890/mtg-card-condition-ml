# clairvoyance
Card condition classifier

# how to train
run train.py

---parameter description---
if you have images in local
	download = False
	img_path = local path of image
else
	download = True
	csv_path = local path of csv which contains image urls and labels
	download_path = local path which you wanna download images to

test_size = split rate (default=0.1)
batch_size = training batch size (default=32)
epochs = number of epoch you wanna train