import tkinter as tk
import tkinter.filedialog as filedialog
from tkinter.scrolledtext import ScrolledText
from os import linesep
from PIL import ImageTk, Image
import threading, queue
from module_dispatcher import dispatch
import threading

mutex_available = True

task_selected = None
image_box = None

window = None
win_height = 400
win_width = 700
selected_path = None
current_path = None
output_box = None
current_mode = None


train_percentage = None
testing_height = None
testing_width = None
testing_epochs = None
output_filename = None


class util:
    @staticmethod
    def get(target):
        if target == "percentage":
            global train_percentage
            return train_percentage.get() 
        elif target == "height":
            global testing_height
            return testing_height.get() 
        elif target == "width":
            global testing_width
            return testing_width.get()
        elif target == "epochs":
            global testing_epochs
            return testing_epochs.get()
        elif target == "filename":
            global output_filename
            return output_filename.get()
        elif target == "path":
            global current_path
            return current_path.get()
        elif target == "mode":
            global current_mode
            return current_mode.get()
        else:
            return "undefined"
    @staticmethod
    def get_folder():
        path = filedialog.askdirectory()
        global  selected_path
        current_path.set(path)
        print(path)
    @staticmethod
    def update_output(str_msg):
        global output_box
        output_box.insert(tk.INSERT, str(str_msg))
        output_box.see(tk.END)
    @staticmethod
    def update_outputln(str_msg):
        global output_box
        output_box.insert(tk.INSERT, str(str_msg)+linesep)
        output_box.see(tk.END)


output_box_msg_queue = queue.Queue()
image_box_img_queue = queue.Queue()
def check_msg_queue():
    global output_box
    global window
    if not output_box_msg_queue.empty():
        util.update_outputln(output_box_msg_queue.get())
    window.after(100,check_msg_queue)
def check_img_queue():
    global image_box
    global window
    if not image_box_img_queue.empty():
        tmp_image = image_box_img_queue.get()
        image_box.configure(image=tmp_image)
        image_box.image = tmp_image
    window.after(100,check_img_queue)


input_lists = ["mode", "percentage", "height", "width", "epochs", "filename", "path"]


string_good = ""
string_bad = ""
string_notify = ""

def good(string_good):
    util.update_outputln("[+] " + string_good)
def bad(string_bad):
    util.update_outputln("[-] " + string_bad)
def notify(string_notify):
    util.update_outputln("[*] " + string_notify)

def go():
    #TODO
    #print("Starts here")
    info_list = []
    for item in input_lists:
        info_list.append(util.get(item))

    global output_box
    global image_box
    global mutex_available
    if not mutex_available:
        bad("Task already running. Please wait.")
        return
    else:
        x = threading.Thread(target=dispatch, args=( mutex_available, info_list[0] , output_box_msg_queue, image_box_img_queue,  info_list))
        x.start()
        #dispatch( mutex_available, info_list[0] , output_box_msg_queue, image_box_img_queue,  info_list)

    pass

def get_window():


    global train_percentage 
    global testing_height 
    global testing_width 
    global testing_epochs 
    global output_filename


    global window
    window = tk.Tk()
    window.title("What\'s that")
    window.geometry(str(win_width)+"x"+str(win_height))
    window.configure(background='white')

    global  output_box
    output_box_content = tk.StringVar()
    output_box_content.set("asdasd")
    output_box = ScrolledText(window, width=50, height=10)
    output_box.grid(row=0, padx=(20,0))

    global image_box
    img_height = 200
    img_width = 200
    place_holder_img = ImageTk.PhotoImage(Image.new('RGB', (img_height, img_width), (255, 255, 255)))
    image_box = tk.Label(window, image=place_holder_img)
    image_box.grid(row=0, column=20, padx=(win_width//2//5,win_width//2//5))

    # user input and hints
    row_index = 1


    hints_train_percentage = tk.Label(window,text="Current Path:")
    hints_train_percentage.grid(row=row_index, column=0, sticky=tk.W)
    #row_index += 1

    global selected_path
    global current_path
    current_path = tk.StringVar()
    current_path.set("./datas")
    selected_path = tk.Entry(window, textvariable=current_path)
    selected_path.grid(row=row_index,column=0)
    row_index += 1


    select_folder = tk.Button(command=util.get_folder, text="select folder")
    select_folder.grid(row=row_index,column=0)
    row_index += 1

#    select_one_image = tk.Button(command=util.get_image, text="select test image")
#    select_one_image.grid(row=row_index,column=0)
#    row_index += 1

    hints_train_percentage = tk.Label(window,text="train percentage(%):")
    hints_train_percentage.grid(row=row_index, column=0, sticky=tk.W)
    #row_index += 1

    global train_percentage
    train_percentage = tk.StringVar()
    train_percentage.set("70")
    train_percentage_box = tk.Entry(window, textvariable=train_percentage)
    train_percentage_box.grid(row=row_index, column=0)
    row_index += 1



    hints_img_height = tk.Label(window, text="sampling height: ")
    hints_img_height.grid(row=row_index,column=0, sticky=tk.W)

    testing_height = tk.StringVar()
    testing_height.set("128")
    testing_height_box = tk.Entry(window, textvariable=testing_height)
    testing_height_box.grid(row=row_index, column=0)
    row_index += 1


    hints_img_width = tk.Label(window, text="sampling width: ")
    hints_img_width.grid(row=row_index,column=0, sticky=tk.W)

    testing_width = tk.StringVar()
    testing_width.set("128")
    testing_width_box = tk.Entry(window, textvariable=testing_width)
    testing_width_box.grid(row=row_index, column=0)
    row_index += 1


    hints_test_epochs = tk.Label(window, text="epochs: ")
    hints_test_epochs.grid(row=row_index,column=0, sticky=tk.W)

    testing_epochs = tk.StringVar()
    testing_epochs.set("3")
    testing_epochs_box = tk.Entry(window, textvariable=testing_epochs)
    testing_epochs_box.grid(row=row_index, column=0)
    row_index += 1



    hints_output_filename = tk.Label(window, text="model path: ")
    hints_output_filename.grid(row=row_index,column=0, sticky=tk.W)

    output_filename = tk.StringVar()
    output_filename.set("Saved_Epochs_30.h5")
    output_filename_box = tk.Entry(window, textvariable=output_filename)
    output_filename_box.grid(row=row_index, column=0)
    row_index += 1

    # end of layout

    row_index = 3
    global current_mode
    current_mode = tk.StringVar(value="train")
    train_button = tk.Radiobutton(window, text="train", variable=current_mode,
                                        indicatoron=False, value="train", width=8)
    test_button = tk.Radiobutton(window, text="test", variable=current_mode,
                                        indicatoron=False, value="test", width=8)
    train_button.grid(row=row_index, column= 20)
    row_index += 1

    test_button.grid(row=row_index, column= 20)
    row_index += 2


    select_folder = tk.Button(command=go, text="Run")
    select_folder.grid(row=row_index,column=20)

    return window




def main():
    global window
    win = get_window()
    check_msg_queue()
    check_img_queue()
    win.mainloop()

    pass

if __name__ == "__main__":
    main()
