# info = ["mode", "percentage", "height", "width", "epochs", "filename", "path"]

msg_queue = None
image_queue = None

def update_outputln(str_msg):
    global msg_queue
    msg_queue.put(str_msg)

def update_image(photoimage):
    global image_queue
    image_queue.put(photoimage)

def notify(string_notify):
    update_outputln("[*] " + string_notify)
def bad(string_bad):
    update_outputln("[-] " + string_bad)
def good(string_bad):
    update_outputln("[+] " + string_bad)


from tensorflow.keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageTk, Image
class callback_plotting(Callback):
 # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    # This function is called at the end of each epoch
    def on_train_batch_end(self, epoch, logs={}):

        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        # Before plotting ensure at least 2 epochs have passed
        #if len(self.losses) > 1:

        N = np.arange(0, len(self.losses))

        # You can chose the style of your preference
        # print(plt.style.available) to see the available options
        #plt.style.use("seaborn")

        # Plot train loss, train acc, val loss and val acc against epochs passed
        plt.figure()
        plt.plot(N, self.losses, label = "train_loss")
        plt.plot(N, self.acc, label = "train_acc")
        #plt.plot(N, self.val_losses, label = "val_loss")
        #plt.plot(N, self.val_acc, label = "val_acc")
        plt.title("Training Loss and Accuracy [Data Batch {}]".format(epoch))
        plt.xlabel("Data Batch #")
        plt.ylabel("Loss/Accuracy")
        #plt.ylabel("Accuracy")
        plt.legend()
        # Make sure there exists a folder called output in the current directory
        # or replace 'output' with whatever direcory you want to put in the plots
        plt.savefig('./.tmp_Epoch-{}.png'.format(epoch))
        plt.close()

        img = Image.open("./.tmp_Epoch-{}.png".format(epoch))
        #img = img.resize((200,200), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(img)
        update_image(photo)


latin_to_chinese = {
 "Paeonia suffruticosa"      : "牡丹",
 "Rosa chinensis"            : "月季",
 "Lilium"                    : "百合",
 "Chrysanthemum morifolium"  : "菊花",
 "Nelumbo nucifera"          : "荷花",
 "Cercis chinensis"  :       "紫荆花",
 "Prunus mume"   :           "梅花",
 "Orchidaceae"   :           "兰花",
 "Dionaea muscipula" :       "食人花",
 "Fritillaria meleagris"  :  "花格贝母 "
}


PRESET_CLASS_NAMES = ["Paeonia suffruticosa", "Rosa chinensis", "Lilium", "Chrysanthemum morifolium", "Nelumbo nucifera", "Cercis chinensis", "Prunus mume", "Orchidaceae", "Dionaea muscipula", "Fritillaria meleagris"]

def test_model(msgqueue, imagequeue, height, width, model_filename, path):

    global image_queue
    global msg_queue
    image_queue = imagequeue
    msg_queue = msgqueue

    height = int(height)
    width = int(width)


    notify("Starting Tensorflow engine...")
    import tensorflow as tf
    notify("Importing Keras...")
    from tensorflow.keras import datasets, layers, models
    from tensorflow.keras.preprocessing import image
    notify("Finishing setup...")
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import pathlib


    notify("Loading model: " +  str(model_filename))
    list_of_accuracies = []
    model = None
    try:
        model = tf.keras.models.load_model(model_filename)
    except:
        bad("Fail to load model.")
        bad("Abort.")
        return None
    #model.signatures["predict"]

    model.summary()

    notify("Model loaded.")

    path = os.path.abspath(path)
    data_dir_path=path
    data_dir = None
    image_count = 0
    notify("Gathering images in current path:")
    all_images = []

    if not os.path.isdir(data_dir_path):
        bad("Directory not available.")
        bad("Abort.")
        return None

    data_dir_path = path
    data_dir = pathlib.Path(data_dir_path)
    for root, dirs, files in os.walk(data_dir_path):
        for f in files:
            if f.endswith(".jpg"):
                all_images.append(os.path.join(root,f))

    index = 0
    test_image = None

    # get testimages !!!
    import time
    for tmp_img_path in all_images:
        test_path = tmp_img_path
        test_img = image.load_img(test_path, target_size=(height,width))
        input_arr = image.img_to_array(test_img)
        input_arr = np.array([input_arr])  # Convert single image to a batch.

        img = Image.open(test_path)
        img = img.resize((400,400), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(img)
        update_image(photo)

        try:
            probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
            predictions = probability_model.predict(input_arr)
        except:
            bad("Error Occurred. Check your logs. Maybe try a different sampling size?")
            break

        result = predictions[0]
        ans = np.argmax(result) 
        notify("Image shown is of class #: " + str(ans))
        notify("In preset, that\'s: " + str(PRESET_CLASS_NAMES[ans]))
        notify("aka: " + latin_to_chinese[PRESET_CLASS_NAMES[ans]])

        for result in predictions:
            print(np.argmax(result)) # answer
            print(result) # all

        time.sleep(2)

        
            



#    for i in all_images:
#        if index > 10:
#            break
#        notify(i)
#
#        #test_image = image.load_img(i, grayscale=False, color_mode='rgb', target_size=(width, height))
#        test_image = image.load_img(i, grayscale=False, color_mode='rgb', target_size=None)
#        test_image = image.img_to_array(test_image)
#        test_image = np.expand_dims(test_image, axis=0)
#        test_image = test_image.reshape(int(width), int(height), 1)
#
#        result = model.predict([test_image,])
#        # .predict vs .predict_classes
#        notify(result)
#
#        index += 1



    #    BATCH_SIZE = 10
    #    IMG_HEIGHT = 64
    #    IMG_WIDTH = 64
    #    STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)


def fit_model(msgqueue, imagequeue, height, width, train_percentage, epochs_count, model_filename, path):

    global image_queue
    global msg_queue
    image_queue = imagequeue
    msg_queue = msgqueue


    notify("Starting Tensorflow engine...")
    import tensorflow as tf
    notify("Importing Keras...")
    from tensorflow.keras import datasets, layers, models
    from tensorflow.keras.preprocessing import image
    notify("Finishing setup...")
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import pathlib
    
    height = int(height)
    width = int(width)
    train_percentage = int(train_percentage)
    epochs_count = int(epochs_count)

    if os.path.isfile(str(model_filename)):
        bad("Model file found: " + str( model_filename))
        bad("Training old model not yet supported.")
        bad("Abort")
        return None
    else:
        notify("Saving to new model file: " + str(model_filename))


    data_dir_path=path
    notify("Obtained the following classes: ")
    CLASS_NAMES = os.listdir(data_dir_path)
    for item in CLASS_NAMES:
        update_outputln(" - " + item)

    data_dir = pathlib.Path(data_dir_path)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    notify("Processing: " + str( image_count)+ " images.")

    train_percentage = abs((100-train_percentage)/100)

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            validation_split=train_percentage,
    horizontal_flip=True
    )


    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        validation_split=train_percentage,
    horizontal_flip=True
    )

    BATCH_SIZE = 10
    IMG_HEIGHT = height
    IMG_WIDTH = width
    STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

    train_generator = train_datagen.flow_from_directory(directory=str(data_dir),
                                                         batch_size=BATCH_SIZE,
                                                         shuffle=True,
                                                         color_mode="rgb",
                                                          class_mode="categorical",
                                                          subset="training",
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         classes = list(CLASS_NAMES))


    validation_generator = valid_datagen.flow_from_directory(directory=str(data_dir),
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=True,
                                                        color_mode="rgb",
                                                        class_mode="categorical",
                                                        subset="validation",
                                                        target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                        classes = list(CLASS_NAMES))



    model = models.Sequential()


    model.add(layers.Conv2D(IMG_HEIGHT, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(IMG_HEIGHT*2, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(IMG_HEIGHT*2, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(IMG_HEIGHT*2, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    num_of_classes = len(CLASS_NAMES)
    #model.add(layers.Dense(10)) # 10 -- # of classes
    model.add(layers.Dense( num_of_classes )) # 10 -- # of classes


    model.compile(optimizer='adam',
                  #loss=tf.keras.losses.categorical_crossentropy,
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  #loss=tf.nn.softmax_cross_entropy_with_logits,
                  metrics=['accuracy'])

#    history = model.fit_generator(
#        train_generator,
#        steps_per_epoch = train_generator.samples // BATCH_SIZE,
#        validation_data = validation_generator,
#        validation_steps = validation_generator.samples // BATCH_SIZE,
#        epochs = epochs_count )


    history = model.fit(
        train_generator,
        steps_per_epoch = train_generator.samples // BATCH_SIZE,
        validation_data = validation_generator,
        validation_steps = validation_generator.samples // BATCH_SIZE,
        epochs = epochs_count,
        callbacks=[callback_plotting()],)

    # Saving trained model
    model.save(model_filename)
    print("Model Saved.")


    #plt.plot(history.history['accuracy'], label='accuracy')
    ##plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    #plt.xlabel('Epoch')
    #plt.ylabel('Accuracy')
    #plt.ylim([0.5, 1])
    #plt.legend(loc='lower right')
    #plt.show()


    test_images, test_labels = validation_generator.next()
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

