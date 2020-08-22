import tkinter as tk
from os import linesep
from module_cnn import test_model, fit_model


from PIL import ImageTk, Image

msg_queue = None
image_queue = None
# info = ["mode", "percentage", "height", "width", "epochs", "filename", "path"]
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

def cleanstr( rawstr ):
    return str(rawstr).strip()

def dispatch( mutex_available, task, msgqueue, imagequeue, info_list ):
    # task == "test"
    # task == "train"
    mutex_available = False # lock task

    global image_queue
    global msg_queue
    image_queue = imagequeue
    msg_queue = msgqueue

    if task not in ["test","train"]:
        bad("Unsupported task. Abort.")
        mutex_available = True
        return None
    if len(cleanstr(info_list[-1])) <= 0:
        bad("Directory not provided. Abort.")
        mutex_available = True
        return None
    if len(cleanstr(info_list[-2])) <= 0:
        bad("Model file not provided. Unable to test. Abort.")
        mutex_available = True
        return None
    if len(cleanstr(info_list[2])) <= 0 or len(cleanstr(info_list[3])) <= 0:
        bad("Dimension incomplete. Abort.")
        mutex_available = True
        return None


    if task == "test":
        notify("Start testing...")
        #def test_model(outputbox, imagebox, height, width, model_filename, path):
        # info = ["mode", "percentage", "height", "width", "epochs", "filename", "path"]
        height = cleanstr(info_list[2])
        width = cleanstr(info_list[3])
        model_filename = cleanstr(info_list[-2])
        testfile_dir = cleanstr(info_list[-1])
        test_model(msg_queue, image_queue, height, width, model_filename, testfile_dir)
        

    elif task == "train":
        if len(cleanstr(info_list[-3])) <= 0:
            bad("Epochs not provided. Unable to train. Abort.")
            mutex_available = True
            return None
        if len(cleanstr(info_list[1])) <= 0:
            bad("Percentage of data used as training data not provided. Abort.")
            mutex_available = True
            return None
        notify("Training your model...")
        place_holder_img = ImageTk.PhotoImage(Image.new('RGB',(200,200),(123,55,123)))
        update_image(place_holder_img)
        #image_box.configure(image=place_holder_img)
        #image_box.image = place_holder_img
        
        #def fit_model(msgqueue, imagequeue, height, width, train_percentage, epochs_count, model_filename, path):
        # info = ["mode", "percentage", "height", "width", "epochs", "filename", "path"]
        height = cleanstr(info_list[2])
        width = cleanstr(info_list[3])
        model_filename = cleanstr(info_list[-2])
        testfile_dir = cleanstr(info_list[-1])
        train_percentage = cleanstr(info_list[1])
        epochs_count = cleanstr(info_list[-3])

        fit_model(msgqueue, imagequeue, height, width, train_percentage, epochs_count, model_filename, testfile_dir)

    mutex_available = True
