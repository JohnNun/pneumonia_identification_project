# Pneumonia Identification Project
by Jonathan Nunez

Pneumonia remains a global health concern, particularly affecting young children, the elderly, and individuals with compromised immune systems. Despite advancements in medicine and science, pneumonia continues to be a leading cause of death accounting for approximately 2.18 million deaths globally in 2021, with the highest impact observed among children under five and adults over 70. While most cases can be treated, early detection in patients can further improve treatment success rate. Through the use of a convolutional neural network model (CNN) and pediatric chest X-ray images, I created a machine learning model with the capability to detect and predict if a patient has pneumonia.

The dataset used for this model was compiled by [Kaggle](https://www.kaggle.com/) and is made up of **5,863 chest X-ray images** sourced from pediatric patients ranging between one to five years of age. The data is categorized into two classes, **“Normal” and “Pneumonia”**. The Data preprocessing involved data augmentation when importing images to lessen model bias and improve generalization as well as data normalization to improve data computation. 

For the model construction I used the TensorFlow and Keras libraries, using the sequential module due to its organization and simple input to output structure. Due to the heavy class imbalance of **4,506 “pneumonia” images and 1,350 “normal” images**, I use methods such as dropout and L2 kernel regularizers to reduce overfitting during the model creation. I also use early stop to stop model training if the model's validation loss increased consecutively rather than decreased.

Among the models created model 5 performed the best overall, with a test data accuracy of 84%. However, the model does have a loss of 0.54, meaning there is still room for improvement. Among other metric results, the model’s precision results in a .86 with “normal” classes and .83 for “pneumonia”. Recall on the other hand shows high sensitivity to “pneumonia” with .93 but a low performance of .69 with the “normal” class.


# Business and Data Understanding
According to the [World Health Organization](https://www.who.int/health-topics/pneumonia/#tab=tab_1), pneumonia is an acute respiratory infection that inflames the air sacs (alveoli) in one or both lungs. It can be caused by bacteria, viruses, or fungi making breathing difficult and reducing oxygen intake. It affects people of all ages but is especially dangerous for young children, older adults, and people with weakened immune systems. Despite the advances in science and medicine, pneumonia remains a major cause of mortality worldwide. In 2021, Pneumonia caused 2.18 million deaths globally, mainly in children younger than 5 years and adults over 70 years, and in those who are susceptible [The Lancet](https://www.thelancet.com/journals/lanres/article/PIIS2213-2600(25)00087-6/fulltext?utm_source). With the development and use of an image classification neural network learning algorithm, it may be possible to detect pneumonia in young children and elderly adults in its early stages possibly leading to prompt treatment.

The data used for this project was retrieved from [Kaggle datasets Chest X-Ray Images](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) this dataset is comprised of 5863 x-ray images divided between two classes, “Normal”, and “Pneumonia”. The chest X-ray images in the dataset were selected from pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center.


# Data Preperation
For data preparation, I first used ImageDataGenerator tool to import all data. I used this tool because it would allow me to add random data augmentation parameters such as flips, set a brightness range, etc. to the training data as its being imported, essentially creating synthetic data and implementing a way to help lower model bias. In the total data, there are 4506 images of pneumonia positive images while there are only 1350 images of pneumonia negative meaning that the data is heavily weighted towards pneumonia positive and as such can lead to severe model bias.

After I've imported the data, I created a separate variable with reshaped data, essentially un-rowing the data to fit the model structural requirements of multiplying the image dimensions and image RGB layers (256 * 256 * 3). To complete my data preparations, I normalize the data bringing the data to a 0 to 1 scale to ensure all features are on a similar scale and improve data computation. I created two sets of normalized data, one with the reshaped data and the other with the original data as CNN models only take in image dimensions.


# Modeling
For the image classification neural network model, I imported tools and libraries from Tensorflow primarily Keras tools. To build the models I used the Sequential module due to its simplicity input to output linear stack model structure. To help with model tuning, I used an early stop method to track the model’s validation loss and stop the model if the validation loss no longer decreased or if it increased consecutively. As for the models learning rate I went with Adaptive Moment Estimation (Adam), as this optimizer uses momentum and RMSprop to automatically adjust the models learning rate.

Due to the data’s heavy class imbalance, the model would often overfit, so to lower overfitting as best as possible I implemented dropout layers to randomly drop nodes in the model. Another method I used to lower overfitting was adding L2 kernel regularizers so to penalize the excessive pneumonia weight from the class imbalance.


# Evaluation
The model that performed the best overall and will be the final model is **model 5, the Convolutional Neural Network model**. Although this model has a rough start, it quickly corrects and starts to learn from the data leading the model to generalize better, resulting in quickly **lowering train and validation loss, and increasing accuracy for training and validation data**.

When evaluating the final model with the test data, this model has an **accuracy of 84% but a loss of 0.54**, so while **the model is confident with its prediction, it appears to be correct about half of the time**. When it comes to the model’s **precision it resulted with 0.86 when it came to identifying normal class, and .83 for the pneumonia class**. For the models recall when identifying the **normal class, it resulted in .69**, while for the pneumonia class the results were **significantly higher with a .93** as the data is **heavily weighted towards the pneumonia class**. Overall, the final model performs well with an 84% accuracy despite the class imbalance and the loss of .54, but **with further tuning or a more balanced dataset I’m sure the model can improve**. 


# Limitations 
A limitation I encountered in this project is the data itself and how **heavily weighted it is towards the pneumonia class as compared to the normal class**. While there are methods that can help alleviate the class weight imbalance and maybe even lessen model bias through the creation of synthetic data, **synthetic data generated from these methods is not as reliable as real data**.

The data gathered from Kaggle datasets while great, it does come already split between train, test, and validation data essentially giving little control over how the data can be divided unless using outside sources or manually moving data.


# Next Steps
For some next steps, further model tweaks and parameter tuning may be necessary to improve the model’s overall performance. **While 84% is not terrible for accuracy with unseen test data, it does leave a little more to be desired especially with a loss of .54** this tells us the model is missing the mark a little more than half of the time.

Another potential next step if time permits would be **testing other Convolutional neural networks such as ResNet, DenseNet or ViT (Vision Transformers) image classification algorithms** and comparing results to the current final model as the base model.

Lastly, the next step I believe can be greatly beneficial to the overall project is **gathering more data**, especially gathering more x-ray images of **healthy pneumonia free lungs**. As stated before the data's **class weights are heavily weighted towards pneumonia positive which can lead to model bias**.


# Conclusion
The model that performed the best overall was model 5, the Convolutional Neural Network model. The final model has an **accuracy of 84% but a loss of 0.54**, so while the model is confident with its prediction it appears to be correct on its decision about half of the time. With that said there can still be room for improvements with the model with **further fine tuning or even by using true data that’s class weight are more balanced**. Overall, the final model is a great start with its predictive capabilities and its high accuracy.


## Repository Structure
* Images
* README.md
* [Presentation](pneumonia_identification_project_presentation.pdf)
* [Notebook](pneumonia_identification_project.ipynb)
