import os
import sys
import time
import random
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import defaultdict
from randaugment import RandAugment

from models import get_model
from datasets import get_dataset
from utils.augment import Cutout
from utils.memory import Memory
from utils.online_sampler import OnlineSampler, OnlineTestSampler
from utils.indexed_dataset import IndexedDataset
from utils.train_utils import select_optimizer, select_scheduler


class _Trainer():
    def __init__(self, *args, **kwargs) -> None:
        self.kwargs = kwargs

        self.mode    = kwargs.get("mode")

        self.n   = kwargs.get("n")
        self.m   = kwargs.get("m")
        self.rnd_NM  = kwargs.get("rnd_NM")
        self.head = kwargs.get("head")

        self.n_tasks = kwargs.get("n_tasks")
        self.dataset_name = kwargs.get("dataset")
        self.rnd_seed    = kwargs.get("rnd_seed")
        self.gpu = kwargs.get("gpu")
        self.gpu_transform = kwargs.get("gpu_transform")
        self.cur_labels = []
        self.memory_size = kwargs.get("memory_size")
        self.log_path    = kwargs.get("log_path")
        self.model_name  = kwargs.get("model_name")
        self.opt_name    = kwargs.get("opt_name")
        self.sched_name  = kwargs.get("sched_name")
        self.batchsize  = kwargs.get("batchsize")
        self.n_worker    = kwargs.get("n_worker")
        self.lr  = kwargs.get("lr")
        self.init_model  = kwargs.get("init_model")
        self.init_opt    = kwargs.get("init_opt")
        self.topk    = kwargs.get("topk")
        self.use_amp = kwargs.get("use_amp")
        self.transforms  = kwargs.get("transforms")
        self.reg_coef    = kwargs.get("reg_coef")
        self.data_dir    = kwargs.get("data_dir")
        self.debug   = kwargs.get("debug")
        self.note    = kwargs.get("note")


        self.selection_size = kwargs.get("selection_size")

        self.eval_period     = kwargs.get("eval_period")
        self.temp_batchsize  = kwargs.get("temp_batchsize")
        self.online_iter     = kwargs.get("online_iter")

        self.imp_update_period   = kwargs.get("imp_update_period")
        

        self.lr_step     = kwargs.get("lr_step")    # for adaptive LR
        self.lr_length   = kwargs.get("lr_length")  # for adaptive LR
        self.lr_period   = kwargs.get("lr_period")  # for adaptive LR

        self.memory_epoch    = kwargs.get("memory_epoch") # for RM
 

        self.start_time = time.time()
        self.num_updates = 0
        self.train_count = 0

        if self.temp_batchsize is None:
            self.temp_batchsize = self.batchsize // 2
        if self.temp_batchsize > self.batchsize:
            self.temp_batchsize = self.batchsize
        self.memory_batchsize = self.batchsize - self.temp_batchsize

        os.makedirs(f"{self.log_path}/logs/{self.dataset_name}/{self.note}", exist_ok=True)

        os.makedirs(f"{self.log_path}/logs/{self.dataset_name}/{self.note}", exist_ok=True)

    def get_device(self, device_id=0):
        if torch.cuda.is_available():
            print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(device_id)}")
            return torch.device(f'cuda:{device_id}')  # 根据传入的设备 ID 选择
        else:
            print("CUDA is not available. Using CPU.")
            return torch.device('cpu')


    def setup_model(self):
        print("Building model...")
        self.model = self.model.to(self.device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.model_without_ddp = self.model
        self.criterion = self.model_without_ddp.loss_fn if hasattr(self.model_without_ddp, "loss_fn") else nn.CrossEntropyLoss(reduction="mean")
        
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)

        n_params = sum(p.numel() for p in self.model_without_ddp.parameters())
        print(f"Total Parameters :\t{n_params}")
        n_params = sum(p.numel() for p in self.model_without_ddp.parameters() if p.requires_grad)
        print(f"Learnable Parameters :\t{n_params}")
        print("")

    def setup_dataset(self):
        # get dataset
        self.train_dataset = self.dataset(root=self.data_dir, train=True, download=True, transform=self.train_transform)
        self.test_dataset = self.dataset(root=self.data_dir, train=False, download=True, transform=self.test_transform)
        self.n_classes = len(self.train_dataset.classes)
        
        self.exposed_classes = []
        
        self.mask = torch.zeros(self.n_classes, device=self.device) - torch.inf
        self.seen = 0
        

    def setup_transforms(self):
        train_transform = []
        self.cutmix = "cutmix" in self.transforms
        if "autoaug" in self.transforms:
            train_transform.append(lambda x: (x*255).type(torch.uint8))
            if 'cifar' in self.dataset_name:
                train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('cifar10')))
            elif 'imagenet' in self.dataset_name:
                train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('imagenet')))
            elif 'svhn' in self.dataset_name:
                train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('svhn')))
            train_transform.append(lambda x: x.type(torch.float32)/255)

        if "cutout" in self.transforms:
            train_transform.append(Cutout(size=16))
        if "randaug" in self.transforms:
            train_transform.append(RandAugment())

        self.test_transform = transforms.Compose([
            transforms.Resize((self.inp_size, self.inp_size), antialias=None),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),])
        
        self.train_transform = transforms.Compose([
                    *train_transform,
                    transforms.Resize((self.inp_size, self.inp_size), antialias=None),
                    transforms.RandomCrop(self.inp_size, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),])


    def run(self) -> None:
        self.device = self.get_device()
        if self.rnd_seed is not None:
            random.seed(self.rnd_seed)
            np.random.seed(self.rnd_seed)
            torch.manual_seed(self.rnd_seed)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed(self.rnd_seed)
                cudnn.deterministic = True
                cudnn.benchmark = False 
                print('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

        print(f"Select a CIL method ({self.mode})")

        self.dataset, self.mean, self.std, self.n_classes = get_dataset(self.dataset_name)

        
        print(f"Building model ({self.model_name})")
        self.model, self.inp_size = get_model(self.kwargs, num_classes = self.n_classes)

        self.setup_transforms()
        self.setup_dataset()
        self.setup_model()
        self.memory = Memory()
        self.total_samples = len(self.train_dataset)

        train_dataset = IndexedDataset(self.train_dataset)
        self.train_sampler = OnlineSampler(train_dataset, self.n_tasks, self.m, self.n, self.rnd_seed, self.rnd_NM, self.selection_size)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batchsize, sampler=self.train_sampler, num_workers=self.n_worker, pin_memory=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batchsize, shuffle=False, num_workers=self.n_worker, pin_memory=True)
        
       
        print(f"Incrementally training {self.n_tasks} tasks")
        task_records = defaultdict(list)
        self.eval_results = defaultdict(list)
        self.samples_cnt = 0
        self.num_eval = self.eval_period
        self.count_class = dict()
        self._cur_task = -1

        
        for task_id in range(self.n_tasks):
            print("Begin train for task" + str(task_id))
            if self.mode == "joint" and task_id > 0:
                return

            print("\n" + "#" * 50)
            print(f"# Task {task_id} Session")
            print("#" * 50 + "\n")
            print("[2-1] Prepare a datalist for the current task")

            self.train_sampler.set_task(task_id)
            self.online_before_task(task_id)
            if self.train_dataloader:
                self.train_learner()  
           
            self.online_after_task(task_id)
            
            test_sampler = OnlineTestSampler(self.test_dataset, self.exposed_classes)
            test_dataloader = DataLoader(self.test_dataset, batch_size=self.batchsize*2, sampler=test_sampler, num_workers=self.n_worker)
            if test_dataloader:
                eval_dict = self.online_evaluate(test_dataloader) 
            
                task_acc = eval_dict['avg_acc']

                print("[2-4] Update the information for the current task")
                task_records["task_acc"].append(task_acc)
                task_records["cls_acc"].append(eval_dict["cls_acc"])

                print("[2-5] Report task result")
        
        
        np.save(f"{self.log_path}/logs/{self.dataset_name}/{self.note}/seed_{self.rnd_seed}.npy", task_records["task_acc"])

        if self.eval_period is not None:
            np.save(f'{self.log_path}/logs/{self.dataset_name}/{self.note}/seed_{self.rnd_seed}_eval.npy', self.eval_results['test_acc'])
            np.save(f'{self.log_path}/logs/{self.dataset_name}/{self.note}/seed_{self.rnd_seed}_eval_time.npy', self.eval_results['data_cnt'])
            
        # Accuracy (A)
        A_auc = np.mean(self.eval_results["test_acc"])
        A_avg = np.mean(task_records["task_acc"])
        A_last = task_records["task_acc"][self.n_tasks - 1]
        
        # Forgetting (F)
        cls_acc = np.array(task_records["cls_acc"])
        F_last = 0.0
        if len(cls_acc.shape) != 1:
            acc_diff = []
            for j in range(self.n_classes):
                if np.max(cls_acc[:-1, j]) > 0:
                    acc_diff.append(np.max(cls_acc[:-1, j]) - cls_acc[-1, j])
            F_last = np.mean(acc_diff)
            np.save(f'{self.log_path}/logs/{self.dataset_name}/{self.note}//seed_{self.rnd_seed}_forget_last.npy', F_last)

        print(f"======== Summary =======")
        print(f"A_auc {A_auc} | A_avg {A_avg} | A_last {A_last} | F_last {F_last}")
        print(f"="*24)


    def train_learner(self):
        eval_dict = dict()
        for i,(images, labels, idx) in enumerate(self.train_dataloader):
            if self.debug and (i+1) * self.temp_batchsize >= 500:
                break
            self.samples_cnt += images.size(0)
            loss, acc = self.online_step(images, labels, idx)
            self.report_training(self.samples_cnt, loss, acc)
            if self.samples_cnt > self.num_eval:
                with torch.no_grad():
                    test_sampler = OnlineTestSampler(self.test_dataset, self.exposed_classes)
                    test_dataloader = DataLoader(self.test_dataset, batch_size=self.batchsize*2, sampler=test_sampler, num_workers=self.n_worker)
                    eval_dict = self.online_evaluate(test_dataloader)
                    self.eval_results["test_acc"].append(eval_dict['avg_acc'])
                    self.eval_results["avg_acc"].append(eval_dict['cls_acc'])        
                    self.eval_results["data_cnt"].append(self.samples_cnt)
                    self.report_test(self.samples_cnt, eval_dict["avg_loss"], eval_dict['avg_acc'])
                    self.num_eval += self.eval_period
            sys.stdout.flush()
        if len(eval_dict)!= 0:
            self.report_test(self.samples_cnt, eval_dict["avg_loss"], eval_dict['avg_acc'])


        

    def add_new_class(self, class_name):
        for label in class_name:
            if label.item() not in self.exposed_classes:
                self.exposed_classes.append(label.item())
            
            self.count_class[label.item()] =  self.count_class.get(label.item(), 0) + 1

        self.memory.add_new_class(cls_list=self.exposed_classes)
        self.mask[:len(self.exposed_classes)] = 0
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)


    def online_step(self, sample, samples_cnt):
        raise NotImplementedError()

    def online_before_task(self, task_id):
        raise NotImplementedError()

    def online_after_task(self, task_id):
        raise NotImplementedError()

    def online_evaluate(self, test_loader, samples_cnt):
        raise NotImplementedError()


    def report_training(self, sample_num, train_loss, train_acc):
        print(
            f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
            f"Num_Classes {len(self.exposed_classes)} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))}"
        )

    def report_test(self, sample_num, avg_loss, avg_acc):
        print(
            f"Test | Sample # {sample_num} | test_loss {avg_loss:.4f} | test_acc {avg_acc:.4f} | "
        )
    
    def _interpret_pred(self, y, pred):
        # xlable is batch
        ret_num_data = torch.zeros(self.n_classes)
        ret_corrects = torch.zeros(self.n_classes)

        xlabel_cls, xlabel_cnt = y.unique(return_counts=True)
        for cls_idx, cnt in zip(xlabel_cls, xlabel_cnt):
            ret_num_data[cls_idx] = cnt

        correct_xlabel = y.masked_select(y == pred)
        correct_cls, correct_cnt = correct_xlabel.unique(return_counts=True)
        for cls_idx, cnt in zip(correct_cls, correct_cnt):
            ret_corrects[cls_idx] = cnt

        return ret_num_data, ret_corrects


    def reset_opt(self):
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)