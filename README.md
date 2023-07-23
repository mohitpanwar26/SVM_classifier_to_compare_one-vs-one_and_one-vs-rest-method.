# SVM_classifier_for_multi-class_classification_to_compare_one-vs-one_and_one-vs-rest-method.

# One Vs One Approach 
  1- The One-vs-One strategy splits a multi-class classification into one binary classification problem per each pair of classes.
  2- N*(N-1)/2  number of SVMs needed.
  3- For example, for 4 classes: ‘red,’ ‘blue,’ and ‘green,’ ‘yellow.’ This could be divided into six binary classification datasets: red vs. blue, red vs. green,         red vs. yellow, blue vs. green, blue vs. yellow, green vs. yellow.

# One Vs Rest Approach
  1- The One-vs-Rest strategy splits a multi-class classification into one binary classification problem per class.
  2- N number of SVMs needed.
  3- For example, for 3 class ‘red,’ ‘blue,’ and ‘green‘. This could be divided into three binary classification datasets: red vs [blue,green], blue vs [red, green],      green vs [red, blue]

# How to run the code :
 
1. Run python file SVM_classifier_ovo_ovr.py.
2. A pop up of gui will be shown.
3. Give the parameter like kernel type and slack parameter value at the bottom of gui.
4. click on input on menu bar.
5. select Image data file and ground truth file provide in Data folder.
6. you can use menu option to visualize its RGB and Grayscale Image.
6. you can save this images using save_RGB_or_GRAYSCALE option.
7. click on output button where you can visualize True_pseudocoloured_image.
8. click on ovo vs ovr option which will show the output of ovo and ovr side by side with its accuracy and time taken for training.
9. click on confusion matrix button to see a accuracy report.
10. for saving output result choose save_output button.
11. You can safely exit using a exit button from a gui.
