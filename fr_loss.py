from keras import backend as K

K.set_image_data_format('channels_first')
from fr_utils import *
from inception_blocks_v2 import *


def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    Implementation of the triplet loss as defined by formula (3)

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # Step 1: Compute the (encoding) distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))
    # Step 2: Compute the (encoding) distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.maximum(tf.reduce_mean(basic_loss), 0.0)

    return loss


FRmodel = faceRecoModel(input_shape=(3, 96, 96))
FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
load_weights_from_FaceNet(FRmodel)
FRmodel.trainable = False


def img_to_encoding_batch(x):
    x_train = []
    for i in range(x.shape[0]):
        # img1 = cv2.imread("E:\\Datasets\\U\\39_0.jpg", 1)
        img1 = x[i]
        img1 = cv2.resize(img1, (96, 96))
        img = img1[..., ::-1]
        img = np.around(np.transpose(img, (2, 0, 1)) / 255.0, decimals=12)
        x_train.append(img)

    x_train = np.array(x_train)
    embedding = FRmodel.predict_on_batch(x_train)
    return embedding


def img_to_encoding(image_path):
    img1 = cv2.imread(image_path, 1)
    img1 = cv2.resize(img1, (96, 96))
    img = img1[..., ::-1]
    img = np.around(np.transpose(img, (2, 0, 1)) / 255.0, decimals=12)
    x_train = np.array([img])
    embedding = FRmodel.predict_on_batch(x_train)
    return embedding


def fr_loss(true, pred):
    encoding_image = img_to_encoding_batch(true)
    encoding_identity = img_to_encoding_batch(pred)

    # dist = np.linalg.norm(encoding_image - encoding_identity, axis=1)
    loss = K.mean(encoding_image - encoding_identity, axis=-1)
    return loss


def verify(image_path, identity):
    """
    Function that verifies if the person on the "image_path" image is "identity".

    Arguments:
    image_path -- path to an image
    identity -- string, name of the person you'd like to verify the identity. Has to be a resident of the Happy house.
    database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
    model -- your Inception model instance in Keras

    Returns:
    dist -- distance between the image_path and the image of "identity" in the database.
    door_open -- True, if the door should open. False otherwise.
    """

    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
    encoding_image = img_to_encoding(image_path)
    encoding_identity = img_to_encoding(identity)

    # Step 2: Compute distance with identity's image (≈ 1 line)
    dist = np.linalg.norm(encoding_image - encoding_identity)

    # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
    if dist < 0.7:
        print("It's " + str(identity) + ", welcome home!")
    else:
        print("It's not " + str(identity) + ", please go away")

    return dist


# verify("E:\\Datasets\\U\\39_0.jpg", "E:\\Datasets\\U\\31_0.jpg", FRmodel)
