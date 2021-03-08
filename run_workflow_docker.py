#!/usr/bin/env python3

"""

MASK DETECTION AND CLASSIFICATION WORKFLOW


"""
import re
import glob,os
import pickle
import logging

from Pegasus.api import *

from utils.wf import split_data_filenames, create_ann_list,create_augmented_filelist


# --- Import Pegasus API ---
from Pegasus.api import *
logging.basicConfig(level=logging.DEBUG)
props = Properties()
# set for checkpointing - if jobs fails or timeouts, Pegasus will retry the job 2 times
# and use the checkpoint to restart the job
props["dagman.retry"] = "1"
props["pegasus.transfer.arguments"] = "-m 1"
props.write()

# DATA AQUSITION
imagesList = glob.glob('data/images/*.png')
annotationList = glob.glob('data/annotations/*.xml')

NUM_TRIALS = 1
NUM_EPOCHS = 1

#DATA SPLIT
train_filenames,val_filenames,test_filenames, files_split_dict = split_data_filenames(imagesList)

#TODO: check the correctness of the fun
train_imgs, train_ann = create_ann_list(train_filenames)
val_imgs, val_ann     = create_ann_list(val_filenames)
test_imgs, test_ann   = create_ann_list(test_filenames)

###################################### REPLICA CATALOG ###########################################################

#add images from group data
rc = ReplicaCatalog()

inputFiles = []
for img in imagesList:
    fileName = img.split("/")[-1]
    img_file = File(fileName)
    inputFiles.append(img_file)
    rc.add_replica("local", img_file,  os.path.join(os.getcwd(),str(img)))
    
annFiles = []
for ann in annotationList:
    fileName = ann.split("/")[-1]
    ann_file = File(fileName)
    annFiles.append(ann_file)
    rc.add_replica("local", ann_file,  os.path.join(os.getcwd(),str(ann)))
    
## add checkpointing file for the hpo model job

def create_pkl(model):
    pkl_filename = "hpo_study_" + model + ".pkl"
    file = open(pkl_filename, 'ab')
    pickle.dump("", file, pickle.HIGHEST_PROTOCOL)
    return pkl_filename

mask_detection_pkl = create_pkl("mask_detection")
mask_detection_pkl_file = File(mask_detection_pkl)
rc.add_replica("local", mask_detection_pkl, os.path.join(os.getcwd(), mask_detection_pkl))

rc.write()


###################################### TRANSFORMATIONS ###########################################################

# Container for all the jobs
tc = TransformationCatalog()
mask_detection_wf_cont = Container(
                "mask_detection_wf",
                Container.DOCKER,
                image="docker://patkraw/mask-detecton-wf:latest",
                arguments="--runtime=nvidia --shm-size=15gb"
            )

tc.add_containers(mask_detection_wf_cont)


dist_plot = Transformation(
                "dist_plot",
                site = "condorpool",
                pfn = os.path.join(os.getcwd(),"bin/plot_class_distribution.py"),
                is_stageable = False,
                container = mask_detection_wf_cont 
            )

augment_imgs = Transformation(
                "augment_images",
                site = "condorpool",
                pfn = os.path.join(os.getcwd(),"bin/data_aug.py"),
                is_stageable = False,
                container = mask_detection_wf_cont 
            )
rename_imgs = Transformation(
                "rename_images",
                site = "condorpool",
                pfn = os.path.join(os.getcwd(),"bin/rename_file.py"),
                is_stageable = False,
                container = mask_detection_wf_cont 
            )

hpo_model = Transformation(
                "hpo_script",
                site = "condorpool",
                pfn = os.path.join(os.getcwd(),"bin/hpo_train.py"),
                is_stageable = False,
                container = mask_detection_wf_cont 
            )

train_model = Transformation(
                "train_script",
                site = "condorpool",
                pfn = os.path.join(os.getcwd(),"bin/train_model.py"),
                is_stageable = False,
                container = mask_detection_wf_cont 
            )


tc.add_transformations(augment_imgs, dist_plot, rename_imgs,hpo_model, train_model)
log.info("writing tc with transformations: {}, containers: {}".format([k for k in tc.transformations], [k for k in tc.containers]))
tc.write()



###################################### CREATE JOBS ###########################################################
wf = Workflow("mask_detection_workflow")

train_preprocessed_files = create_augmented_filelist(train_filenames,2)
distribution_plot_file = File("class_distribution.png")
val_preprocessed_files = [File("val_"+ f.split("/")[-1]) for f in val_filenames]
test_preprocessed_files = [File("test_"+ f.split("/")[-1]) for f in test_filenames]

# DATA EXPLORATION
# takes in all the annotationa files and creates plot with distribution of the classes
distribution_plot_job = Job(dist_plot)
distribution_plot_job.add_inputs(*train_ann, *val_ann, *test_ann)
distribution_plot_job.add_outputs(distribution_plot_file)

# DATA PREPROCESSING:TRAIN DATA-DATA AUGMENTATION
# takes images and adds gaussian noise to them
preprocess_train_job = Job(augment_imgs)
preprocess_train_job.add_inputs(*train_imgs)
preprocess_train_job.add_outputs(*train_preprocessed_files)

# DATA PREPROCESSING:VAL DATA-FILE RENAMING
preprocess_val_job = Job(rename_imgs)
preprocess_val_job.add_inputs(*val_imgs)
preprocess_val_job.add_outputs(*val_preprocessed_files)
preprocess_val_job.add_args("val")

# DATA PREPROCESSING:TEST DATA-FILE RENAMING
preprocess_test_job = Job(rename_imgs)
preprocess_test_job.add_inputs(*test_imgs)
preprocess_test_job.add_outputs(*test_preprocessed_files)
preprocess_test_job.add_args("test")

# HYPERPARAMETER OPTIMIZATION

hpo_job = Job(hpo_model)
hpo_job.add_args("--epochs",NUM_EPOCHS, "--trials", NUM_TRIALS)
hpo_job.add_inputs(*train_preprocessed_files,*train_ann,*val_preprocessed_files,*val_ann)
hpo_job.add_outputs(File("best_hpo_params.txt"))
hpo_job.set_stdout("output_hpo_job.txt")
hpo_job.add_checkpoint(mask_detection_pkl_file, stage_out=True)
hpo_job.add_profiles(Namespace.PEGASUS, key="maxwalltime", value=10)


# MODEL TRAINING
#model_training_job = Job(train_script)
#model_training_job.add_inputs(*train_imgs)
#model_training_job.add_outputs(*train_preprocessed_files)
#model_training_job.add_checkpoint(fastRCNNP_pkl_file, stage_out=True)
#model_training_job.add_profiles(Namespace.PEGASUS, key="maxwalltime", value=2)


# MODEL EVALUATION


# INFERENCE
# takes images of our labmates and classifies them




###################################### RUN WORKFLOW ###########################################################
def main():
	wf.add_jobs(
	    distribution_plot_job,
	    preprocess_train_job,
	    preprocess_val_job,
	    preprocess_test_job,
	 #   hpo_job
	)

	try:
	    wf.plan(submit=True)
	    wf.wait()
	    wf.statistics()
	except PegasusClientError as e:
	    print(e.output)


if __name__ == "__main__":
    main()
