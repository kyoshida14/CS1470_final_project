import tensorflow as tf
import numpy as np
import pathlib 


def imageLoader(imDir):
	image_root = pathlib.Path(imDir)
	listDs = tf.data.Dataset.list_files(str(image_root/'*.png'))
	imageList = None
	for f in listDs:
		image = tf.io.decode_png( tf.io.read_file(f) )
		image = tf.image.resize(image, [256,256])  #change to width height channels dont resize
		#image = tf.reshape(image, [1,-1])
		image = tf.expand_dims(image, axis = 0)

		if imageList == None:
			imageList = image
		else:
			imageList = tf.concat([imageList, image], axis = 0) 


	return imageList
		

def readData(realDir, fakeDir):
	
	realImages =  (imageLoader(realDir) / 255 - 0.449) / 0.226
	fakeImages =  (imageLoader(fakeDir) / 255 - 0.449) / 0.226

	return((realImages, fakeImages)) 