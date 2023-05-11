# Pegasus Workflow for Mask Detection and Classification

The following project addresses the problem of determining what percentage of the population is properly wearing masks to better track our collective efforts in preventing the spread of COVID-19 in public spaces. To help solve this problem, we leverage modern deep learning tools such as the Optuna hyper parameter optimization framework and the [FastRCNNPredictor](https://arxiv.org/abs/1506.01497) model. The experiment is organized as a scientific workflow and utilizes the Pegasus Workflow Management System to handle its execution on distributed resources. 


The workflow uses **images of masks on faces** and **annotations** related to each image as classified as one of the following **three categories** as the main input dataset:
* wearing a mask 
* not wearing a mask
* wearing a mask incorrectly

The dataset is split into training, validation, and test sets before the workflow starts.  The **Pre-processing** step and Data Augmentation of Images is done to resize images and normalize them to make sure the data is consistent among all classes and also to avoid class imbalance. Additionally, **image augmentation** is done by injecting Gaussian noise. Next, the train and validation data are passed to the **hyperparameter optimization** step, where different learning rates are explored. The **training** of **FastRCNN** model is done with the recommended learning rate on the concatenated train and validation set, and obtains the weights. Then the **evaluation** is performed test set in order to generate a txt file with the scores for relevant performance metrics like average running loss. Finally, **predictions** can be made with any user input images using the trained model and show mask detection results.

**Machine Learning steps in the workflow :**
<br>
<img src="imgs/ml_steps3.png" style="width: 900px;"/>
<br>


### Currently implemented jobs in Pegasus
![workflow](imgs/mask_detection.png)


### Running the Workflow

* Clone the respository using the command `git clone <repository link>`
* `cd` into the `mask-detection-workflow` directory
*  [Optional] If you want to add your own docker image, go to `run_workflow_docker.py` file and change the image in 

    ```python
    mask_detection_wf_cont = Container(
                Container.SINGULARITY,
                image="docker://zaiyancse/mask-detection:latest",
                image_site="docker_hub"
            )
    ``` 
    
  part, to the link to your docker image
* Run the workflow script using the command `python3 run_workflow_docker.py`
* Check 

### Overview of the Workflow
<img src="imgs/MaskDetection.png" />

### Detail view of jobs in the Workflow
![workflow](imgs/wf_graph2.png)
<img src="imgs/mask_dectection_wf2.png" />


### Results
<img src="imgs/prediction.png" />
<img src="imgs/cm.png" style="width: 400px/><br>
<img src="imgs/group_mask_detection.png" />

### Authors
Project by Kelsie Lam, research intern.<br>
Updated by Zaiyan Alam.
