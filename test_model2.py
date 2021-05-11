import numpy as np
from keras.preprocessing import image
from keras.models import load_model

#LOADING MODEL
model = load_model('Trained_model2.h5')
# model.evaluate()

#LOADING IMAGE
image_path = './data/test_set/D/1.png'
test_img = image.load_img(image_path, target_size=(28,28))
test_img = image.img_to_array(test_img)
test_img = np.expand_dims(test_img,axis=0)
result = model.predict(test_img)

print('Predicted sign is:')

if result[0][0] == 1:
	print('A')
elif result[0][1] == 1:
	print('B')
elif result[0][2] == 1:
	print('C')
elif result[0][3] == 1:
	print('D')
elif result[0][4] == 1:
	print('E')
elif result[0][5] == 1:
	print('F')
elif result[0][6] == 1:
	print('G')
elif result[0][7] == 1:
	print('H')
elif result[0][8] == 1:
	print('I')
elif result[0][9] == 1:
	print('J')
elif result[0][10] == 1:
	print('K')
elif result[0][11] == 1:
	print('L')
elif result[0][12] == 1:
	print('M')
elif result[0][13] == 1:
	print('N')
elif result[0][14] == 1:
	print('O')
elif result[0][15] == 1:
	print('P')
elif result[0][16] == 1:
	print('Q')
elif result[0][17] == 1:
	print('R')
elif result[0][18] == 1:
	print('S')
elif result[0][19] == 1:
	print('T')
elif result[0][20] == 1:
	print('U')
elif result[0][21] == 1:
	print('V')
elif result[0][22] == 1:
	print('W')
elif result[0][23] == 1:
	print('X')
elif result[0][24] == 1:
	print('Y')
elif result[0][25] == 1:
	print('Z')
