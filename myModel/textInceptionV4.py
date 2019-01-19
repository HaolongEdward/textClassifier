
def text_inception_base(num_unique_token, feature_dimension, MAX_SEQUENCE_LENGTH):
	# feature_dimension = 64
	cnn = Sequential()
	# cnn.add(InputLayer(MAX_SEQUENCE_LENGTH))

	cnn.add(Embedding(num_unique_token+1, feature_dimension, input_length=MAX_SEQUENCE_LENGTH))
	
	


def text_inception(num_classes):
