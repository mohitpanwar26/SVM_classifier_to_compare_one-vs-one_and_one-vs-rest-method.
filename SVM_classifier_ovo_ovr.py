import os,time
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from tkinter import filedialog
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
import numpy as np
import tkinter as tk
from tkinter import *
from matplotlib.figure import Figure

def UploadAction1():
    global data,data_name
    # Importing image data file:
    data = loadmat(filedialog.askopenfilename(initialdir=os.getcwd(), title='Select Image Data File'))
    data_name=next(reversed(data))
    data=data[data_name]

def UploadAction2():
    global label,X_train_scaled,X_test_scaled,X_total_scaled,y,y_train,label_name,y_test
    # Importing ground truth file:
    label = loadmat(filedialog.askopenfilename(initialdir=os.getcwd(), title='Select Ground Truth File'))
    label_name=next(reversed(label))
    label=label[label_name]
    # Flatten the data to 2D array
    X = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
    y = label.ravel()
    # Split the dataset into training and testing sets
    X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.3, random_state=42)
    # Perform Min-Max scaling on input features
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_total_scaled = scaler.fit_transform(X)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

def showgrayscale():
    global image_output
    # Choose a band for grayscale display
    band1 = 29  # Choose band 
    # Display the grayscale image
    image_grayscale = data[:, :, band1]
    # # Normalize the Grayscale image data to [0, 255]
    image_grayscale = (image_grayscale - np.min(image_grayscale)) / (np.max(image_grayscale) - np.min(image_grayscale)) * 255
    image_grayscale = image_grayscale.astype(np.uint8)
    # view image on gui:
    from PIL import ImageTk, Image
    img = None
    img1=None
    lbl2=Label(root)
    lbl2.pack()
    image_grayscale = Image.fromarray(image_grayscale)
    image_output = image_grayscale.resize((900,600))
    # to show output image
    img1 =ImageTk.PhotoImage(image_output)
    lbl2.configure(imag=img1)
    lbl2.imag=img1
    # Defining clear function:
    def clear():
        lbl2.configure(imag=None)
        lbl2.imag=None
        clear_button.destroy()
        lbl2.destroy()
    # Add a button to clear the image result from the screen
    clear_button = Button(root, text='  CLEAR  ', command=clear,bg='red',fg='blue',font=('Arial', 12))
    clear_button.pack(pady=10)

def showRGB():
    global image_output
    # Choose three bands for RGB display
    band1 = 29  # Choose band 29
    band2 = 19  # Choose band 19
    band3 = 9  # Choose band 9
    # Extract the selected bands from the image data
    image_rgb = np.stack((data[:, :, band1], data[:, :, band2], data[:, :, band3]), axis=-1)
    # # Normalize the RGB image data to [0, 255]
    image_rgb = (image_rgb - np.min(image_rgb)) / (np.max(image_rgb) - np.min(image_rgb)) * 255
    image_rgb = image_rgb.astype(np.uint8)
    # view image on gui:
    from PIL import ImageTk, Image
    img = None
    img1=None
    lbl2=Label(root)
    lbl2.pack()
    image_rgb = Image.fromarray(image_rgb)
    image_output = image_rgb.resize((900,600))
    # to show output image
    img1 =ImageTk.PhotoImage(image_output)
    lbl2.configure(imag=img1)
    lbl2.imag=img1
    # Defining clear function :
    def clear():
        lbl2.configure(imag=None)
        lbl2.imag=None
        # clear_button.pack_forget()
        clear_button.destroy()
        lbl2.destroy()

    # Add a button to clear the image result from the screen
    clear_button = Button(root, text='  CLEAR  ', command=clear,bg='red',fg='blue',font=('Arial', 12))
    clear_button.pack(pady=10)

def showpseudocolortrue():
    global fig
    # Define the class labels
    label_mapping ={'indian_pines_gt':{0: 'Undefined',1: 'Alfalfa',2: 'Corn-notill',3: 'Corn-mintill',4: 'Corn',5: 'Grass-pasture',
            6: 'Grass-trees',7: 'Grass-pasture-mowed',8: 'Hay-windrowed',9: 'Oats',10: 'Soybean-notill',11: 'Soybean-mintill',
            12: 'Soybean-clean',13: 'Wheat',14: 'Woods',15: 'Buildings-Grass-Trees-Drives',16: 'Stone-Steel-Towers',},
            'salinasA_gt':{0: 'Undefined',1:'Brocoli_green_weeds_1',2:'Brocoli_green_weeds_2',3:'Fallow',4:'Fallow_rough_plow',
            5:'Fallow_smooth',6:'Stubble',7:'Celery',8:'Grapes_untrained',9:'Soil_vinyard_develop',10:'Corn_senesced_green_weeds',
            11:'Lettuce_romaine_4wk',12:'Lettuce_romaine_5wk',13:'Lettuce_romaine_6wk',14:'Lettuce_romaine_7wk',15:'Vinyard_untrained',
            16:'Vinyard_vertical_trellis'},
            'salinas_gt':{0: 'Undefined',1:'Brocoli_green_weeds_1',2:'Brocoli_green_weeds_2',3:'Fallow',4:'Fallow_rough_plow',
            5:'Fallow_smooth',6:'Stubble',7:'Celery',8:'Grapes_untrained',9:'Soil_vinyard_develop',10:'Corn_senesced_green_weeds',
            11:'Lettuce_romaine_4wk',12:'Lettuce_romaine_5wk',13:'Lettuce_romaine_6wk',14:'Lettuce_romaine_7wk',15:'Vinyard_untrained',
            16:'Vinyard_vertical_trellis'}}
    # Define the pseudocolor map
    cmap = plt.get_cmap('jet', len(label_mapping[label_name]))
    # Set the color of the "Undefined" class to white
    cmap.set_under('white')
    # Create the pseudocolor image
    fig, ax = plt.subplots()
    im = ax.imshow(label, cmap=cmap,vmin=0.5)
    # Create the legend
    # Define the color for the "Undefined" class as white
    undefined_color = 'white'
    colors = [cmap(i) if i != 0 else undefined_color for i in range(len(label_mapping[label_name]))]
    patches = [plt.Rectangle((0,0),1,1,fc=colors[i]) for i in range(len(label_mapping[label_name]))]
    legend = ax.legend(patches, [label_mapping[label_name][i] for i in label_mapping[label_name]], loc='lower left', bbox_to_anchor=(1.1, -0.005), ncol=1,title='Class Labels')
    # Display the image and legend
    plt.title(f'{data_name} Pseudocolored True Label Image')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    # Add a button to remove the canvas from the GUI
    remove_button = tk.Button(root, text='  CLEAR  ', command=lambda: (canvas.get_tk_widget().destroy(), remove_button.pack_forget()),bg='red',fg='blue',font=('Arial', 12))
    remove_button.pack()
  

def ovoandovrclassification():
    global fig
    global y_pred_ovr,y_test_ovr,training_time_ovr,Time_taken_ovr
    global y_pred_ovo,y_test_ovo,training_time_ovo,Time_taken_ovo
    # Getting user defined kernel_type and slack parameter:
    kernel=kernel_type.get()
    C=float(slack_given.get())

    # Define the SVM model with non-linear kernel and varying slack parameter C
    svm_model = SVC(kernel=kernel, C=C)

    # Fit the SVM model with the one versus one method
    start_time = time.time()
    svm_ovo = OneVsOneClassifier(svm_model).fit(X_train_scaled, y_train)
    Time_taken_ovo = time.time() - start_time
    # Predict the labels for test data:
    y_test_ovo = svm_ovo.predict(X_test_scaled) 
    # testing on whole dataset for visualization
    y_pred_ovo = svm_ovo.predict(X_total_scaled) 

    # Fit the SVM model with the one versus rest method
    start_time1 = time.time()
    svm_ovr = OneVsRestClassifier(svm_model).fit(X_train_scaled, y_train)
    Time_taken_ovr = time.time() - start_time1
    # Predict the labels for test data:
    y_test_ovr = svm_ovr.predict(X_test_scaled) 
   
    # testing on whole dataset for visualization
    y_pred_ovr = svm_ovr.predict(X_total_scaled)

    # Reshaping :
    y_champ_ovo=y_pred_ovo.reshape(data.shape[0],data.shape[1])
    y_champ_ovr=y_pred_ovr.reshape(data.shape[0],data.shape[1])

    from sklearn.metrics import accuracy_score
    acc_ovo = accuracy_score(y_test, y_test_ovo)
    acc_ovr = accuracy_score(y_test, y_test_ovr)

    # Print the results
    print("Accuracy (One-vs-One): {:.2f}%".format(acc_ovo * 100))
    print("Accuracy (One-vs-Rest): {:.2f}%".format(acc_ovr * 100))
    # Print the time taken for one-vs-one strategy
    print("Training time (One-vs-One): {:.2f} seconds".format(Time_taken_ovo))
    # Print the time taken one-vs-rest strategy
    print("Training time (One-vs-Rest): {:.2f} seconds".format(Time_taken_ovr))

    # Define the class labels
    label_mapping ={'indian_pines_gt':{0: 'Undefined',1: 'Alfalfa',2: 'Corn-notill',3: 'Corn-mintill',4: 'Corn',5: 'Grass-pasture',
                6: 'Grass-trees',7: 'Grass-pasture-mowed',8: 'Hay-windrowed',9: 'Oats',10: 'Soybean-notill',11: 'Soybean-mintill',
                12: 'Soybean-clean',13: 'Wheat',14: 'Woods',15: 'Buildings-Grass-Trees-Drives',16: 'Stone-Steel-Towers',},
                'salinasA_gt':{0: 'Undefined',1:'Brocoli_green_weeds_1',2:'Brocoli_green_weeds_2',3:'Fallow',4:'Fallow_rough_plow',
                5:'Fallow_smooth',6:'Stubble',7:'Celery',8:'Grapes_untrained',9:'Soil_vinyard_develop',10:'Corn_senesced_green_weeds',
                11:'Lettuce_romaine_4wk',12:'Lettuce_romaine_5wk',13:'Lettuce_romaine_6wk',14:'Lettuce_romaine_7wk',15:'Vinyard_untrained',
                16:'Vinyard_vertical_trellis'},
                'salinas_gt':{0: 'Undefined',1:'Brocoli_green_weeds_1',2:'Brocoli_green_weeds_2',3:'Fallow',4:'Fallow_rough_plow',
                5:'Fallow_smooth',6:'Stubble',7:'Celery',8:'Grapes_untrained',9:'Soil_vinyard_develop',10:'Corn_senesced_green_weeds',
                11:'Lettuce_romaine_4wk',12:'Lettuce_romaine_5wk',13:'Lettuce_romaine_6wk',14:'Lettuce_romaine_7wk',15:'Vinyard_untrained',
                16:'Vinyard_vertical_trellis'}}

    # Define the pseudocolor map for one vs one classification:
    cmap = plt.get_cmap('jet', len(label_mapping[label_name]))
    # Set the color of the "Undefined" class to white
    cmap.set_under('white')
    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # Create the pseudocolor image for ovo classification
    im1 = ax1.imshow(y_champ_ovo, cmap=cmap,vmin=0.5)
    # Create the pseudocolor image for ovr classification
    im2 = ax2.imshow(y_champ_ovr, cmap=cmap,vmin=0.5)
    # Set the titles for the subplots
    ax1.set_title(f'{data_name} Pseudocolored ovo Label Image'+'\nAccuracy : {:.2f}%'.format(acc_ovo * 100)+
                  "\nTraining time : {:.2f} seconds.".format(Time_taken_ovo))
    ax2.set_title(f'{data_name} Pseudocolored ovr Label Image'+'\nAccuracy : {:.2f}%'.format(acc_ovr * 100)+
                  "\nTraining time : {:.2f} seconds.".format(Time_taken_ovr))
    # Create the legend for the entire figure
    # Define the color for the "Undefined" class as white
    undefined_color = 'white'
    colors = [cmap(i) if i != 0 else undefined_color for i in range(len(label_mapping[label_name]))]
    patches = [plt.Rectangle((0,0),1,1,fc=colors[i]) for i in range(len(label_mapping[label_name]))]
    labels = [label_mapping[label_name][i] for i in label_mapping[label_name]]
    legend=ax1.legend(patches, labels, loc='upper center', bbox_to_anchor=(1.1, 0), ncol=8,title='Class Labels')
    legend.get_frame().set_edgecolor('black')
    legend.get_title().set_fontsize(12)
    for text in legend.get_texts():
        text.set_fontsize(8)
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Add a button to remove the canvas from the GUI
    remove_button = tk.Button(root, text='  CLEAR  ', command=lambda: (canvas.get_tk_widget().destroy(), remove_button.pack_forget()),bg='red',fg='blue',font=('Arial', 12))
    remove_button.pack()

def accuracyreport():
    global fig
    # Calculate confusion matrix for one-vs-one strategy
    confusion_matrix_ovo = np.zeros((label.max(), label.max()), dtype=int)
    for i in range(len(y_test)):
        confusion_matrix_ovo[y_test[i]-1][y_test_ovo[i]-1] += 1
    # Calculate confusion matrix for one-vs-rest strategy
    confusion_matrix_ovr = np.zeros((label.max(), label.max()), dtype=int)
    for i in range(len(y_test)):
        confusion_matrix_ovr[y_test[i]-1][y_test_ovr[i]-1] += 1

    # Calculate overall accuracy for one-vs-one and one-vs-rest strategies
    total = np.sum(confusion_matrix_ovo)
    correct_ovo = np.sum(np.diag(confusion_matrix_ovo))
    overall_acc_ovo = correct_ovo / total
    correct_ovr = np.sum(np.diag(confusion_matrix_ovr))
    overall_acc_ovr = correct_ovr / total

    # Calculate user's accuracy, producer's accuracy, omission error, and commission error for one-vs-one strategy
    user_acc_ovo = np.zeros((label.max(),))
    prod_acc_ovo = np.zeros((label.max(),))
    omission_error_ovo = np.zeros((label.max(),))
    commission_error_ovo = np.zeros((label.max(),))
    for i in range(label.max()):
        user_acc_ovo[i] = confusion_matrix_ovo[i,i] / np.sum(confusion_matrix_ovo[i,:])
        prod_acc_ovo[i] = confusion_matrix_ovo[i,i] / np.sum(confusion_matrix_ovo[:,i])
        omission_error_ovo[i] = 1 - user_acc_ovo[i]
        commission_error_ovo[i] = 1 - prod_acc_ovo[i]

    # Calculate user's accuracy, producer's accuracy, omission error, and commission error for one-vs-rest strategy
    user_acc_ovr = np.zeros((label.max(),))
    prod_acc_ovr = np.zeros((label.max(),))
    omission_error_ovr = np.zeros((label.max(),))
    commission_error_ovr = np.zeros((label.max(),))
    for i in range(label.max()):
        user_acc_ovr[i] = confusion_matrix_ovr[i,i] / np.sum(confusion_matrix_ovr[i,:])
        prod_acc_ovr[i] = confusion_matrix_ovr[i,i] / np.sum(confusion_matrix_ovr[:,i])
        omission_error_ovr[i] = 1 - user_acc_ovr[i]
        commission_error_ovr[i] = 1 - prod_acc_ovr[i]

    print(f'overall Accuracy (one vs one) : {overall_acc_ovo}')
    print(f'overall Accuracy (one vs rest) : {overall_acc_ovr}')
    print(f'user\'s accuracy (one vs one) : {user_acc_ovo}')
    print(f'Producer\'s accuracy (one vs one) : {prod_acc_ovo}')
    print(f'user\'s accuracy (one vs rest) : {user_acc_ovr}')
    print(f'Producer\'s accuracy (one vs rest) : {user_acc_ovr}')
    print(f'Omission error (one vs one) : {omission_error_ovo}')
    print(f'Commission error (one vs one) : {commission_error_ovo}')
    print(f'Omission error (one vs rest) : {omission_error_ovr}')
    print(f'Commission error (one vs rest) : {commission_error_ovr}')

    # Create 1x2 subplot
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    # Plot confusion matrix for one-vs-one strategy
    sns.heatmap(confusion_matrix_ovo, annot=True, fmt='d', cmap='Blues', ax=axs[0])
    axs[0].set_xlabel('Predicted Class')
    axs[0].set_ylabel('True Class')
    axs[0].set_title('Confusion Matrix (One-vs-One)')
    # Plot confusion matrix for one-vs-rest strategy
    sns.heatmap(confusion_matrix_ovr, annot=True, fmt='d', cmap='Blues', ax=axs[1])
    axs[1].set_xlabel('Predicted Class')
    axs[1].set_ylabel('True Class')
    axs[1].set_title('Confusion Matrix (One-vs-Rest)')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Add a button to remove the canvas from the GUI
    remove_button = tk.Button(root, text='  CLEAR  ', command=lambda: (canvas.get_tk_widget().destroy(), remove_button.pack_forget()),bg='red',fg='blue',font=('Arial', 12))
    remove_button.pack()

def save_output():
    # Allow the user to choose the output path and filename
    output_path = filedialog.asksaveasfilename(defaultextension='.png')
    if output_path:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Save the figure to the chosen path and filename
        fig.savefig(output_path)

def save_RGB_or_GRAYSCALE():
    # Allow the user to choose the output path and filename
    output_path = filedialog.asksaveasfilename(defaultextension='.png')
    if output_path:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Save the figure to the chosen path and filename
        image_output.save(output_path)

# Building a GUI :
root = tk.Tk()
root.title('support vector machine:')

#for adding menu bar:
Menubar = Menu(root)
FileMenu = Menu(Menubar, tearoff=0)
FileMenu.add_command(label='Image Data File ', command=UploadAction1)
FileMenu.add_separator()
FileMenu.add_command(label='Ground Truth File', command=UploadAction2)
Menubar.add_cascade(label="Input",menu=FileMenu)
root.config(menu=Menubar)

FileMenu1 = Menu(Menubar, tearoff=0)
FileMenu1.add_command(label='     RGB Image    ', command=showRGB)
FileMenu1.add_separator()
FileMenu1.add_command(label=' Gray-Scale Image ', command=showgrayscale)
Menubar.add_cascade(label="Menu",menu=FileMenu1)
root.config(menu=Menubar)

FileMenu2 = Menu(Menubar, tearoff=0)
FileMenu2.add_command(label='True pseudocolor Image ', command=showpseudocolortrue)
FileMenu2.add_separator()
FileMenu2.add_command(label='  OVO VS OVR   ', command=ovoandovrclassification)
FileMenu2.add_separator()
FileMenu2.add_command(label='Confusion Matrix', command=accuracyreport)
Menubar.add_cascade(label="Output",menu=FileMenu2)
root.config(menu=Menubar)

FileMenu3 = Menu(Menubar, tearoff=0)
FileMenu3.add_command(label='save_RGB_or_GRAYSCALE', command=save_RGB_or_GRAYSCALE)
FileMenu3.add_separator()
FileMenu3.add_command(label='save_output', command=save_output)
Menubar.add_cascade(label="Save",menu=FileMenu3)
root.config(menu=Menubar)

FileMenu4 = Menu(Menubar, tearoff=0)
FileMenu4.add_command(label="Exit", command=quit)
Menubar.add_cascade(label="Exit",menu=FileMenu4)
root.config(menu=Menubar)

# creating label
from tkinter import ttk
c_p = Label(root, text = "Enter the slack parameter (C): ",font=('Arial', 12))
c_p.place(x = 0, y = 650)
C_value= tk.DoubleVar()
label = tk.Label(root, text="Choose a kernel type:",font=('Arial', 12))
label.place(x=0,y=623)

# Create a combobox
options = ["linear", "poly", "rbf", "sigmoid"]
kernel_type = tk.StringVar()
combo_box = ttk.Combobox(root, textvariable=kernel_type, values=options)
combo_box.place(x=220, y=620)

# creating entry box
slack_given = Entry(root, textvariable = C_value)
slack_given .place(x=220,y=653)
root.geometry("1000x700")
root.mainloop()