from os import pathsep
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch
import torch.nn as nn
import torchvision.models as models

import collections

class clothing_dataset(Dataset): 
    def __init__(self, root, transform, mode, num_samples=0, pred=[], probability=[], paths=[], num_class=14, \
        labeled_balance=False, eval_selection=False, target_loader_size=None): 
        
        self.root = root
        self.transform = transform
        self.mode = mode
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}            

        self.num_class = num_class

        self.eval_selection = eval_selection
        self.target_loader_size = target_loader_size
        
        with open('%s/noisy_label_kv.txt'%self.root,'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()           
                img_path = '%s/'%self.root+entry[0][7:]
                self.train_labels[img_path] = int(entry[1])                         
        with open('%s/clean_label_kv.txt'%self.root,'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()           
                img_path = '%s/'%self.root+entry[0][7:]
                self.test_labels[img_path] = int(entry[1])   

        if mode == 'all':
            train_imgs=[]
            with open('%s/noisy_train_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root+l[7:]
                    train_imgs.append(img_path)                                
            random.shuffle(train_imgs)
            class_num = torch.zeros(num_class)
            self.train_imgs = []
            for impath in train_imgs:
                label = self.train_labels[impath] 
                if class_num[label]<(num_samples/14) and len(self.train_imgs)<num_samples:
                    self.train_imgs.append(impath)
                    class_num[label]+=1
            random.shuffle(self.train_imgs)       
        elif self.mode == "labeled" or self.mode == 'labeled_eval':   
            if self.eval_selection:
                pred, probability, paths = self.selection(pred, probability, paths, loadertype='labeled')
            train_imgs = paths 
            pred_idx = pred.nonzero()[0]
            self.train_imgs = [train_imgs[i] for i in pred_idx]                
            self.probability = [probability[i] for i in pred_idx]            

            if labeled_balance:
                label_list = [self.train_labels[i] for i in self.train_imgs]
                min_num_class = min(collections.Counter(label_list).values())
                new_train_imgs = list()
                new_probability = list()
                for c in range(num_class):
                    _ = [i for i in range(len(self.train_imgs)) if self.train_labels[self.train_imgs[i]] == c]
                    _ = _[:min_num_class]
                    new_train_imgs += [self.train_imgs[i] for i in _]
                    new_probability += [self.probability[i] for i in _]
                self.train_imgs = new_train_imgs
                self.probability = new_probability

            print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))
        elif self.mode == "unlabeled":  
            if self.eval_selection:
                pred, probability, paths = self.selection(pred, probability, paths, loadertype='unlabeled')
            train_imgs = paths 
            pred_idx = (1-pred).nonzero()[0]  
            self.train_imgs = [train_imgs[i] for i in pred_idx]                
            self.probability = [probability[i] for i in pred_idx]            
            print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))                                    
                         
        elif mode=='test':
            self.test_imgs = []
            with open('%s/clean_test_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root+l[7:]
                    self.test_imgs.append(img_path)            
        elif mode=='val':
            self.val_imgs = []
            with open('%s/clean_val_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root+l[7:]
                    self.val_imgs.append(img_path)
                    
    def __getitem__(self, index):  
        if self.mode=='labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path] 
            prob = self.probability[index]
            image = Image.open(img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2, target, prob              
        elif self.mode=='labeled_eval':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path] 
            image = Image.open(img_path).convert('RGB')    
            img1 = self.transform(image) 
            return img1, target, img_path
        elif self.mode=='unlabeled':
            img_path = self.train_imgs[index]
            image = Image.open(img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2  
        elif self.mode=='all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image)
            return img, target, img_path        
        elif self.mode=='test':
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target
        elif self.mode=='val':
            img_path = self.val_imgs[index]
            target = self.test_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target    
        
    def __len__(self):
        if self.mode=='test':
            return len(self.test_imgs)
        if self.mode=='val':
            return len(self.val_imgs)
        else:
            return len(self.train_imgs)            
    
    def selection(self, predlist, problist, pathlist, loadertype):
        origin_size = predlist.shape[0]
        target_size = self.target_loader_size
        if loadertype=='labeled':
            target_labeled_size = target_size/origin_size * predlist.sum()
            true_indices = predlist.nonzero()[0]
        elif loadertype=='unlabeled':
            target_labeled_size = target_size/origin_size * (1-predlist).sum()
            true_indices = (1-predlist).nonzero()[0]
        np.random.shuffle(true_indices)
        true_indices = true_indices[:int(target_labeled_size)]
        pred, prob = predlist[true_indices], problist[true_indices]
        new_pathlist = []
        for index in true_indices:
            new_pathlist.append(pathlist[index])
        return pred, prob, new_pathlist

    def selection_by_cls(self, predlist, problist, pathlist, target_size):
        noisy_labels = np.array([self.train_labels[p_] for p_ in pathlist])
        origin_size = predlist.shape[0]
        target_labeled_size = target_size / origin_size * predlist.sum()
        target_labeled_size_cls = target_labeled_size / 14
        target_unlabeled_size = target_size / origin_size * (1 - predlist).sum()
        target_unlabeled_size_cls = target_unlabeled_size / 14

        labeled_indices = []
        unlabeled_indices = []
        for cls_ in range(14):
            labeled_indices_cls = ((noisy_labels == cls_) & predlist).nonzero()[0]
            unlabeled_indices_cls = ((noisy_labels == cls_) & ~predlist).nonzero()[0]
            np.random.shuffle(labeled_indices_cls)
            np.random.shuffle(unlabeled_indices_cls)
            labeled_indices.append(labeled_indices_cls[:int(target_labeled_size_cls)])
            unlabeled_indices.append(unlabeled_indices_cls[:int(target_unlabeled_size_cls)])
        labeled_indices = np.concatenate(labeled_indices)
        unlabeled_indices = np.concatenate(unlabeled_indices)
        selected_indices = np.concatenate([labeled_indices, unlabeled_indices])
        np.random.shuffle(selected_indices)
        new_predlist = predlist[selected_indices]
        new_problist = problist[selected_indices]
        new_pathlist = [pathlist[_] for _ in selected_indices]

        assert new_predlist.shape[0] == new_problist.shape[0] == len(new_pathlist)
        return new_predlist, new_problist, new_pathlist
        
class clothing_dataloader():  
    def __init__(self, root, batch_size, num_batches, num_workers):    
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_batches = num_batches
        self.root = root
                   
        self.transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),                
                transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),                     
            ]) 
        self.transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
            ])        
    def run(self,mode,pred=[],prob=[],paths=[], labeled_balance=False, balance_compensate=False, warmup_batches_num=None, num_batches_eval=None, eval_selection=None, gpu_num=1):        
        assert not balance_compensate
        if mode=='warmup':
            if warmup_batches_num is not None:
                warmup_dataset = clothing_dataset(self.root,transform=self.transform_train, mode='all',num_samples=warmup_batches_num*self.batch_size*2)
            else:
                warmup_dataset = clothing_dataset(self.root,transform=self.transform_train, mode='all',num_samples=self.num_batches*self.batch_size*2)
            warmup_loader = DataLoader(
                dataset=warmup_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)  
            return warmup_loader
        elif mode=='train':
            labeled_dataset = clothing_dataset(self.root,transform=self.transform_train, mode='labeled',pred=pred, probability=prob,paths=paths, \
                labeled_balance=labeled_balance, eval_selection=eval_selection, \
                target_loader_size=self.num_batches * self.batch_size)
            labeled_loader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)           
            unlabeled_dataset = clothing_dataset(self.root,transform=self.transform_train, mode='unlabeled',pred=pred, probability=prob,paths=paths, \
                eval_selection=eval_selection, target_loader_size=self.num_batches*self.batch_size)
            unlabeled_loader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=int(self.batch_size),
                shuffle=True,
                num_workers=self.num_workers)   
            return labeled_loader,unlabeled_loader
        elif mode == 'train_eval':
            labeled_dataset = clothing_dataset(self.root,transform=self.transform_train, mode='labeled_eval',pred=pred, probability=prob,paths=paths, \
                labeled_balance=labeled_balance, eval_selection=eval_selection, \
                target_loader_size=self.num_batches * self.batch_size)
            labeled_loader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size * gpu_num,
                shuffle=True,
                num_workers=self.num_workers)           
            unlabeled_dataset = clothing_dataset(self.root,transform=self.transform_train, mode='unlabeled',pred=pred, probability=prob,paths=paths, \
                eval_selection=eval_selection, target_loader_size=self.num_batches*self.batch_size)
            unlabeled_loader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=int(self.batch_size) * gpu_num,
                shuffle=True,
                num_workers=self.num_workers)   
            return labeled_loader,unlabeled_loader
        elif mode=='eval_train':
            if num_batches_eval is None:
                eval_dataset = clothing_dataset(self.root,transform=self.transform_test, mode='all',num_samples=self.num_batches*self.batch_size, noisy_sample_mix=False)
                eval_loader = DataLoader(
                    dataset=eval_dataset, 
                    batch_size=self.batch_size * gpu_num,
                    shuffle=False,
                    num_workers=self.num_workers)          
                return eval_loader        
            else:
                eval_dataset = clothing_dataset(self.root,transform=self.transform_test, mode='all',num_samples=num_batches_eval*self.batch_size)
                eval_loader = DataLoader(
                    dataset=eval_dataset, 
                    batch_size=self.batch_size * gpu_num,
                    shuffle=False,
                    num_workers=self.num_workers)          
                return eval_loader        
        elif mode=='test':
            test_dataset = clothing_dataset(self.root,transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=512,
                shuffle=False,
                num_workers=self.num_workers)             
            return test_loader             
        elif mode=='val':
            val_dataset = clothing_dataset(self.root,transform=self.transform_test, mode='val')
            val_loader = DataLoader(
                dataset=val_dataset, 
                batch_size=512,
                shuffle=False,
                num_workers=self.num_workers)             
            return val_loader     
