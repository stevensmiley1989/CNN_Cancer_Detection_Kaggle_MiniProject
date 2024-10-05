from ultralytics import YOLO
import os
import ray
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air import session
import datetime

from ray.tune.schedulers import ASHAScheduler
cwd = os.getcwd()
date_i = str(datetime.datetime.now()).replace(" ","_").replace(".","p")

'''
asha_scheduler:
  * metric: is for the metric to schedule with
  * mode: the max is used to maximize the metric
  * max_t: 100 is used as the maximum number of trials
  * grace_period: This is for how many trials to go without terminating if metric is not improving
  * 
'''
asha_scheduler = ASHAScheduler(
    metric="metrics/accuracy_top1",    
    mode="max",              
    max_t=100,                 
    grace_period=2,            
    reduction_factor=2         
)
'''
ray.init:
    * num_gpus: set to 1 so I can use my Orin's gpu
    * num_cpus: set to 8 so I can use some of my Orin's cpu resources without using all 12
    * object_store_memory: set to 10**9 to ensure memory is allocated
    * dashboard_host: for viewing in browser
    * dashboard_port: for viewing in browser
'''
ray.init(num_gpus=1, num_cpus=8, object_store_memory=10**9,dashboard_host="0.0.0.0", dashboard_port=8265)

def model_train(config):
    '''
    model_train:
        * archs:  Different YOLOv8 classifier architectures for training
        * MAX_NUM: A unique value from the Jupyter Notebook that created the directories for training
        * project: The unique location the trained model will be stored during training
        * data: The path to the training data
    '''
    archs = ["yolov8n-cls.pt","yolov8l-cls.pt"]
    MAX_NUM = 100000000
    for arch_i in archs:
        arch_j = arch_i.split(".")[0]
        project = os.path.join(cwd, f"RAYTUNE_MODELS_NEW_{date_i}", arch_j)
        if not os.path.exists(project):
            os.makedirs(project)
        model = YOLO(arch_i)
        name_i = str(datetime.datetime.now()).replace(" ","_").replace(".","p")
        data = f"/mnt/DATA1/DTSA_5011/Week3_Assignment/yolo_data_{MAX_NUM}"
        
        momentum = config["momentum"]
        batch = config["batch_size"]
        optimizer = config["optimizer"]
        mixup = config["mixup"]
        mosaic = config["mosaic"]
        fliplr = config["fliplr"]
        flipud = config["flipud"]
        
        stopping_threshold = 0.6  
        prev = 0.0
        cnt = 0
        patience = 2

        MAX_EPOCHS = 40
        for epoch in range(MAX_EPOCHS):
            name_j = name_i + f"_EPOCH_{epoch}_"

            model.train(
                data=data,
                epochs=1,  
                batch=batch,
                imgsz=128,
                patience=patience,
                momentum=momentum,
                optimizer=optimizer,
                fliplr=fliplr,
                flipud=flipud,
                mosaic=mosaic,
                mixup=mixup,
                project=project,
                name = name_j,
            )
            
            eval_results = model.val()

            top1_accuracy = eval_results.results_dict.get('metrics/accuracy_top1', 0)
            if top1_accuracy>prev:
                prev = top1_accuracy
                cnt = 0
            else:
                cnt +=1

            print(f"Epoch {epoch}: Evaluation Results:", eval_results.results_dict)
            
            session.report({"metrics/accuracy_top1": top1_accuracy})
            
            if top1_accuracy < stopping_threshold or cnt>patience:
                print(f"Stopping trial {session.get_trial_id()} early due to low accuracy: {top1_accuracy}")
                return  # Early exit to stop this specific trial


flipuds = [0.0, 0.5]
fliplrs = [0.0, 0.5]
mosaics = [0.5, 1.0]
mixups = [0.0, 0.1, 1.0]

search_space = {
    "batch_size": tune.choice([32, 16]),
    "momentum": tune.loguniform(0.7, 0.98),
    "optimizer": tune.choice(['SGD', 'AdamW']),
    "flipud": tune.choice(flipuds),
    "fliplr": tune.choice(fliplrs),
    "mixup": tune.choice(mixups),
    "mosaic": tune.choice(mosaics),
}


search_alg = HyperOptSearch(metric="metrics/accuracy_top1", mode="max")

num_samples = 30
tuner = tune.Tuner(
    tune.with_resources(model_train, resources={"cpu": 8, "gpu": 1}),
    tune_config=tune.TuneConfig(
        search_alg=search_alg,
        num_samples=num_samples,
        scheduler=asha_scheduler, 
    ),
    param_space=search_space,
)
results = tuner.fit()

ray.shutdown()
