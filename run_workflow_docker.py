import re
import glob,os
import pickle
import logging
from pathlib import Path
from utils.wf import split_data_filenames, create_ann_list,create_augmented_filelist

logging.basicConfig(level=logging.DEBUG)

# --- Import Pegasus API -----------------------------------------------------------
from Pegasus.api import *

# --- Top Directory Setup ----------------------------------------------------------
top_dir = Path(__file__).parent.resolve()

# DATA AQUSITION
imagesList = glob.glob('data/images/*.png')
predict_images = glob.glob('data/images/pred_imgs/*.png')
annotationList = glob.glob('data/annotations/*.xml')

NUM_TRIALS = 1
NUM_EPOCHS = 1

#DATA SPLIT
train_filenames,val_filenames,test_filenames, files_split_dict = split_data_filenames(imagesList)

#ANNOTATIONS
train_imgs, train_ann = create_ann_list(train_filenames)
val_imgs, val_ann     = create_ann_list(val_filenames)
test_imgs, test_ann   = create_ann_list(test_filenames)

######################################## PROPERTIES ###########################################################
props = Properties()
props["dagman.retry"] = "1"
props["pegasus.mode"] = "development"
props.write()


###################################### REPLICA CATALOG ###########################################################

rc = ReplicaCatalog()

inputFiles = []
for img in imagesList:
    fileName = img.split("/")[-1]
    img_file = File(fileName)
    inputFiles.append(img_file)
    rc.add_replica("local", img_file,  os.path.join(os.getcwd(),str(img)))
    
pred_imgs = []
for img in predict_images:
    fileName = img.split("/")[-1]
    img_file = File(fileName)
    pred_imgs.append(img_file)
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

fastRCNNP_pkl = create_pkl("fastRCNNP")
fastRCNNP_pkl_file = File(fastRCNNP_pkl)
rc.add_replica("local", fastRCNNP_pkl, os.path.join(os.getcwd(), fastRCNNP_pkl))

rc.write()

###################################### TRANSFORMATIONS ###########################################################

# Container for all the jobs
tc = TransformationCatalog()
mask_detection_wf_cont = Container(
                "mask_detection_wf",
                Container.SINGULARITY,
                image="docker://zaiyancse/mask-detection:latest",
                image_site="docker_hub"
            )

tc.add_containers(mask_detection_wf_cont)


dist_plot = Transformation(
                "dist_plot",
                site = "local",
                pfn = top_dir/"bin/plot_class_distribution.py",
                is_stageable = True,
                container = mask_detection_wf_cont 
            )

augment_imgs = Transformation(
                "augment_images",
                site = "local",
                pfn = top_dir/"bin/data_aug.py",
                is_stageable = True,
                container = mask_detection_wf_cont 
            )

rename_imgs = Transformation(
                "rename_images",
                site = "local",
                pfn = top_dir/"bin/rename_file.py",
                is_stageable = True,
                container = mask_detection_wf_cont 
            )

hpo_model = Transformation(
                "hpo_script",
                site = "local",
                pfn = top_dir/"bin/hpo_train.py",
                is_stageable = True,
                container = mask_detection_wf_cont 
            )

train_model = Transformation(
                "train_script",
                site = "local",
                pfn = top_dir/"bin/train_model.py",
                is_stageable = True,
                container = mask_detection_wf_cont 
            )

evaluate_model = Transformation(
                "evaluate_script",
                site = "local",
                pfn = top_dir/"bin/evaluate.py",
                is_stageable = True,
                container = mask_detection_wf_cont 
            )

predict_detection = Transformation(
                "predict_script",
                site = "local",
                pfn = top_dir/"bin/predict.py",
                is_stageable = True,
                container = mask_detection_wf_cont 
            )


tc.add_transformations(augment_imgs, dist_plot, rename_imgs, hpo_model, train_model, evaluate_model, predict_detection)
logging.info("writing tc with transformations: {}, containers: {}".format([k for k in tc.transformations], [k for k in tc.containers]))
tc.write()

###################################### CREATE JOBS ###########################################################
wf = Workflow("mask_detection_workflow")

train_preprocessed_files = create_augmented_filelist(train_filenames,2)
distribution_plot_file = File("class_distribution.png")
val_preprocessed_files = [File("val_"+ f.split("/")[-1]) for f in val_filenames]
test_preprocessed_files = [File("test_"+ f.split("/")[-1]) for f in test_filenames]

distribution_plot_job = Job(dist_plot)
distribution_plot_job.add_args(distribution_plot_file)
distribution_plot_job.add_inputs(*train_ann, *val_ann, *test_ann)
distribution_plot_job.add_outputs(distribution_plot_file)
wf.add_jobs(distribution_plot_job)

# TRAIN DATA AUGMENTATION
preprocess_train_job = Job(augment_imgs)
preprocess_train_job.add_inputs(*train_imgs)
preprocess_train_job.add_outputs(*train_preprocessed_files,stage_out=False)
wf.add_jobs(preprocess_train_job)

# VAL DATA-FILE RENAMING
preprocess_val_job = Job(rename_imgs)
preprocess_val_job.add_inputs(*val_imgs)
preprocess_val_job.add_outputs(*val_preprocessed_files,stage_out=False)
preprocess_val_job.add_args("val")
wf.add_jobs(preprocess_val_job)

# TEST DATA-FILE RENAMING
preprocess_test_job = Job(rename_imgs)
preprocess_test_job.add_inputs(*test_imgs)
preprocess_test_job.add_outputs(*test_preprocessed_files,stage_out=False)
preprocess_test_job.add_args("test")
wf.add_jobs(preprocess_test_job)

hpo_params = File("best_hpo_params.txt")
hpo_job = Job(hpo_model)
hpo_job.add_args("--epochs",NUM_EPOCHS, "--trials", NUM_TRIALS,"--results_file",hpo_params)
hpo_job.add_inputs(*train_preprocessed_files,*train_ann,*val_preprocessed_files,*val_ann)
hpo_job.add_outputs(hpo_params)
hpo_job.add_checkpoint(mask_detection_pkl_file, stage_out=True)
wf.add_jobs(hpo_job)

model_file = File("mask_detection_model.pth")
model_training_job = Job(train_model)
model_training_job.add_args(hpo_params,model_file)
model_training_job.add_inputs(hpo_params,*train_imgs,
                              *train_preprocessed_files, *val_preprocessed_files,
                              *test_preprocessed_files, *annFiles)
model_training_job.add_checkpoint(fastRCNNP_pkl_file, stage_out=True)
model_training_job.add_outputs(model_file)
wf.add_jobs(model_training_job)

confusion_matrix_file = File("confusion_matrix.png")
evaluation_file = File("evaluation.txt")
model_evaluating_job = Job(evaluate_model)
model_evaluating_job.add_args(model_file,evaluation_file,confusion_matrix_file)
model_evaluating_job.add_inputs(model_file,*test_preprocessed_files, *annFiles)
model_evaluating_job.add_outputs(evaluation_file,confusion_matrix_file)
wf.add_jobs(model_evaluating_job)

predicted_image = File("predicted_image.png")
predicted_classes = File("predictions.txt")
predict_detection_job.add_args(model_file,predicted_image,predicted_classes)
predict_detection_job.add_inputs(model_file,*pred_imgs, *annFiles)
predict_detection_job.add_outputs(predicted_image,predicted_classes)
wf.add_jobs(predict_detection_job)

try:
    wf.plan(submit=True)
except PegasusClientError as e:
    print(e.output)
    
    
