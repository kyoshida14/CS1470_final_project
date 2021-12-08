import tensorflow as tf
import numpy as np
import pathlib


def imageLoader(imDir, startIndices):
	image_root = pathlib.Path(imDir)
	listDs = tf.data.Dataset.list_files(str(image_root/'*.png'), shuffle=False)
	imageList = None
	indexIncrement = 1000
	

	listDs = list(listDs.as_numpy_iterator())
	if startIndices == None: #set to none if all images can be read
		indexIncrement = len(listDs)
		startIndices = 0 #[startIndices : startIndices+indexIncrement]
	for f in listDs[startIndices: startIndices+indexIncrement]:
		#print(f)
		image = tf.io.decode_png( tf.io.read_file(f) )
		image = tf.image.resize(image, [256,256])  #change to width height channels dont resize
		#image = tf.reshape(image, [1,-1])
		image = tf.expand_dims(image, axis = 0)

		if imageList == None:
			imageList = image
		else:
			imageList = tf.concat([imageList, image], axis = 0)


	return imageList


def readData(realDir, fakeDir, startIndices):

	realImages =  (imageLoader(realDir, startIndices) / 255 - 0.449) / 0.226
	fakeImages =  (imageLoader(fakeDir, startIndices) / 255 - 0.449) / 0.226

	return((realImages, fakeImages))


def readData1(dir):
	images =   (imageLoader(dir, None) / 255 - 0.449) / 0.226
	return images
