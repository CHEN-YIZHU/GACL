
from torch.utils.data import DataLoader
from methods._trainer import _Trainer
import torch
import itertools as it
from torch.nn import functional as F
from tqdm import tqdm
import torchvision.transforms as transforms
import sys
from utils.online_sampler import OnlineTestSampler
from utils.utils import AverageMeter
import torch.distributed as dist
import os



class ACIL(_Trainer):
    def __init__(self, **kwargs):
        super(ACIL, self).__init__(**kwargs)
        self.exposed_classes = []

        self.dtype = torch.double
        self.out_features = 0
        self.feature_size = self.hidden
        

        self.W = torch.zeros((self.feature_size, 0)).double().cuda()

        # Autocorrelation Memory Matrix
        self.R = (torch.eye(self.feature_size) / self.Gamma).double().cuda()
        

    def train_learner(self):
        eval_dict = dict()
        self.model.eval()
        for ep in range(self.epoch):
            if self.train_dataloader:
                # for i,(images, labels, idx) in enumerate(self.train_dataloader):
                for images, labels, idx in self.train_dataloader:
                
                    self.samples_cnt += (images.size(0)) * self.world_size

                    # self.samples_cnt += (images.size(0)) * self.world_size
                    loss, acc = self.online_step(images, labels, idx)

                    self.report_training(self.samples_cnt, loss, acc)

                    if self.samples_cnt > self.num_eval:
                        with torch.no_grad():
                            test_sampler = OnlineTestSampler(self.test_dataset, self.exposed_classes)
                            test_dataloader = DataLoader(self.test_dataset, batch_size=self.batchsize*2, sampler=test_sampler, num_workers=self.n_worker)
                            eval_dict = self.online_evaluate(test_dataloader)
                            if self.distributed:
                                eval_dict =  torch.tensor([eval_dict['avg_loss'], eval_dict['avg_acc'], *eval_dict['cls_acc']], device=self.device)
                                dist.reduce(eval_dict, dst=0, op=dist.ReduceOp.SUM)
                                eval_dict = eval_dict.cpu().numpy()
                                eval_dict = {'avg_loss': eval_dict[0]/self.world_size, 'avg_acc': eval_dict[1]/self.world_size, 'cls_acc': eval_dict[2:]/self.world_size}
                            if self.is_main_process():  
                                self.eval_results["test_acc"].append(eval_dict['avg_acc'])
                                self.eval_results["avg_acc"].append(eval_dict['cls_acc'])
                                self.eval_results["data_cnt"].append(self.samples_cnt)
                                self.eval_results["exposed_class"].append(len(self.exposed_classes))
                                self.report_test(self.samples_cnt, eval_dict["avg_loss"], eval_dict['avg_acc'])
                            self.num_eval += self.eval_period
                    sys.stdout.flush()


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X_fe = self.model.expansion(X).double().cuda()
        return X_fe @ self.W

    def online_step(self, X, y, idx):
        """Use the data form data_loader to train the classifier incrementally."""
        self.add_new_class(y) 
        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())
        loss = AverageMeter()
        _loss, _acc, _iter = 0.0, 0.0, 0
        for _ in range(int(self.online_iter)):
            X = X.to(self.device)
            y = y.to(self.device)
            # X = self.train_transform(X)
            self.fit(X, y)
            logits = self.forward(X)
            _, pred_label = torch.max(logits, 1)
            acc = (pred_label == y).sum().item() / y.size(0) #  correct_cnt
            _loss += loss.avg()
            _acc += acc
            _iter += 1
        
        return _loss / _iter, _acc / _iter

    @torch.no_grad()
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Train the classifier incrementally by the input features X and label y (integers, not one-hot)"""
        
        X_fe = self.model.expansion(X).double().cuda()
        # GCIL
        num_classes = max(self.out_features, torch.max(y).item() + 1)
        assert isinstance(num_classes, int)
        if num_classes > self.out_features:
            increment_size = num_classes - self.out_features
            tail = torch.zeros((self.W.shape[0], increment_size)).to(self.W)
            self.W = torch.concat((self.W, tail), dim=1)
            self.out_features = num_classes

        Y = F.one_hot(y, self.out_features).double().cuda()
        # print("Y1", Y.shape)
        # 假设 Y 是一个 n*d 维度的张量
        n, d = Y.size()
        # print("old", self.old_num)
        zeros_tensor = torch.zeros(n, self.old_num).to(self.device)
        Y = torch.cat((zeros_tensor, Y[:, self.old_num:]), dim=1)
        self.R = self.R.cuda()
        self.W = self.W.cuda()
        K = torch.eye(X_fe.shape[0]).to(X_fe) + X_fe @ self.R @ X_fe.T
        self.R -= self.R @ X_fe.T @ torch.inverse(K) @ X_fe @ self.R
        self.W += self.R @ X_fe.T @ (Y - X_fe @ self.W)

        

        assert torch.isfinite(K).all().item(),      "Pay attention to the numerical stability."
        assert torch.isfinite(self.R).all().item(), "Pay attention to the numerical stability."
        assert torch.isfinite(self.W).all().item(), "Pay attention to the numerical stability."

    def online_evaluate(self, test_loader):
        total_correct, total_loss = 0.0, 0.0
        total_num_data = 0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, y = data

                x = x.to(self.device)
                y = y.to(self.device)
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())
                logit = self.forward(x)

                pred = torch.argmax(logit, dim=-1)
                
                _, pred_label = logit.topk(self.topk, 1, True, True)


                total_correct += torch.sum(pred_label == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                
                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                label += y.tolist()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        eval_dict = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}
        
        return eval_dict

    
    def online_after_task(self, task_id):
        pass

    def online_before_task(self, task_id):
        self.old_num = len(self.exposed_classes)
        pass
