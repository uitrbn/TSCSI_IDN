from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import optimizer
import torchvision
import torchvision.models as models
import random
import argparse
import numpy as np
from PreResNet import *
from sklearn.mixture import GaussianMixture
import dataloader_cifar_idn_part as dataloader
from tqdm import tqdm
from skimage.filters import threshold_otsu

from utils import cifar_config, parser_bool

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--lr_other', '--learning_rate_other', default=0.04, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.3, type=float, help='clean probability threshold')
parser.add_argument('--d_threshold', default=0.3, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./cifar-10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--idn_noise', default=0.6, type=float)
parser.add_argument('--min_discrepancy_weight', default=0.1, type=float)
parser.add_argument('--max_discrepancy_weight', default=0.1, type=float)
parser.add_argument('--dr_dim', default=128, type=int)
parser.add_argument('--warmup_epoch', default=30, type=int)
parser.add_argument('--dis_iter', default=100, type=int)
args = parser.parse_args()
args.id = __file__

cifar_config(args)

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def create_model_1fc():
    class ResClassifier(nn.Module):
        def __init__(self, class_num=14):
            super(ResClassifier, self).__init__()
            self.fc1 = nn.Sequential(
                nn.BatchNorm1d(args.dr_dim, affine=True),
                nn.ReLU(inplace=True),
                nn.Linear(args.dr_dim,  class_num)
                )

        def forward(self, x):
            logit = self.fc1(x)
            return logit
    class MyModel(nn.Module):
        def __init__(self, num_classes1, num_classes2):
            super(MyModel, self).__init__()
            assert num_classes1 == num_classes2
            self.num_classes = num_classes1
            self.model_resnet = resnet_cifar34(num_classes=num_classes1)
            num_ftrs = self.model_resnet.linear.in_features
            self.model_resnet.linear = nn.Identity()

            self.classification_fc = nn.Linear(num_ftrs, num_classes1)

            self.dr = nn.Linear(num_ftrs, args.dr_dim)
            self.fc1 = ResClassifier(num_classes1)
            self.fc2 = ResClassifier(num_classes1)

        def forward(self, x, detach_feature=False):
            feature = self.model_resnet(x)
            res_out = self.classification_fc(feature)

            if detach_feature:
                feature = feature.detach()
            dr_feature = self.dr(feature)
            out1 = self.fc1(dr_feature)
            out2 = self.fc2(dr_feature)
            output_mean = (out1 + out2) / 2
            return feature, res_out, dr_feature, out1, out2, output_mean

    model = MyModel(args.num_class, args.num_class)
    model = model.cuda()
    return model     

# Training
def train(epoch,net,net2,optimizer_res,labeled_trainloader,unlabeled_trainloader, optimizer_other):
    net.train()
    net2.eval()
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()                 
        batch_size = inputs_x.size(0)
        if batch_size == 1 or inputs_u.size(0) == 1:
            continue
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            _, outputs_u11, _, _, _, _ = net(inputs_u)
            _, outputs_u12, _, _, _, _ = net(inputs_u2)
            _, outputs_u21, _, _, _, _ = net2(inputs_u)
            _, outputs_u22, _, _, _, _ = net2(inputs_u2)    
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            _, outputs_x, _, _, _, _ = net(inputs_x)
            _, outputs_x2, _, _, _, _ = net(inputs_x2)
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        _, logits, _, output_d1, output_d2, _ = net(mixed_input, detach_feature=True)

        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]        
           
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + lamb * Lu  + penalty

        logits_x_d1 = output_d1[:batch_size*2]
        logits_u_d1 = output_d1[batch_size*2:]
        logits_x_d2 = output_d2[:batch_size*2]
        logits_u_d2 = output_d2[batch_size*2:]
        Lx_other_d1, Lu_other_d1, lamb_other_d1 = criterion(logits_x_d1, mixed_target[:batch_size*2], logits_u_d1, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
        Lx_other_d2, Lu_other_d2, lamb_other_d2 = criterion(logits_x_d2, mixed_target[:batch_size*2], logits_u_d2, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
        loss_other = Lx_other_d1 + lamb_other_d1 * Lu_other_d1 + Lx_other_d2 + lamb_other_d2 * Lu_other_d2

        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean_d1 = torch.softmax(output_d1, dim=1).mean(0)
        pred_mean_d2 = torch.softmax(output_d2, dim=1).mean(0)
        penalty_other = torch.sum(prior*torch.log(prior/pred_mean_d1)) + torch.sum(prior*torch.log(prior/pred_mean_d2))
        penalty_other = penalty_other / 2
        loss_other = loss_other + penalty_other

        optimizer_res.zero_grad()
        loss.backward()
        optimizer_res.step()
        optimizer_other.zero_grad()
        loss_other.backward()
        optimizer_other.step()

        _, _, _, output_d1, output_d2, _ = net(mixed_input)
        L_discrepancy_other = get_discrepancy(output_d1, output_d2, mixed_target)
        loss_dis = L_discrepancy_other.mean() * args.min_discrepancy_weight

        optimizer_res.zero_grad()
        loss_dis.backward()
        optimizer_res.step()  
        
        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                %(args.dataset, args.idn_noise, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
        sys.stdout.flush()

def warmup(epoch, net,optimizer_res,dataloader,optimizer_other):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer_res.zero_grad()
        _, res_out, _, output1, output2, _ = net(inputs)
        loss = CEloss(res_out, labels)      
        if args.noise_mode=='asym':  # penalize confident prediction for asymmetric noise
            penalty = conf_penalty(res_out)
            L = loss + penalty      
        elif args.noise_mode=='sym':   
            L = loss
        L.backward()  
        optimizer_res.step() 

        optimizer_other.zero_grad()
        _, res_out, _, output1, output2, _ = net(inputs)
        loss = CEloss(output1, labels) + CEloss(output2, labels)
        L = loss
        L.backward()  
        optimizer_other.step() 

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.idn_noise, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()

def test_net_mean(epoch,net1,net2):
    net1.eval()
    net2.eval()
    res_correct = 0
    dmean_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, outputs1, _, _, _, output_mean_a = net1(inputs)
            _, outputs2, _, _, _, output_mean_b = net2(inputs)        
            res_out = outputs1 + outputs2
            output_mean = output_mean_a + output_mean_b

            _, res_predicted = torch.max(res_out, 1)
            _, dmean_predicted = torch.max(output_mean, 1)
            total += targets.size(0)
            res_correct += res_predicted.eq(targets).cpu().sum().item()
            dmean_correct += dmean_predicted.eq(targets).cpu().sum().item()
                       
    res_acc = 100. * res_correct / total
    dmean_acc = 100. * dmean_correct / total
    print("Epoch: %d | 2Net Test Acc: Res: %.2f%%  dmean: %.2f%%" %(epoch, res_acc, dmean_acc))  
    return res_acc

def test_epoch(net, test_loader):
    net.eval()
    res_correct = 0
    dmean_correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader), ncols=0)
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.cuda(), targets.cuda()
            _, res_out, _, _, _, output_mean = net(inputs)
            _, res_predicted = torch.max(res_out, 1)
            _, dmean_predicted = torch.max(output_mean, 1)
            total += targets.size(0)
            res_correct += res_predicted.eq(targets).cpu().sum().item()
            dmean_correct += dmean_predicted.eq(targets).cpu().sum().item()
    res_acc = 100. * res_correct / total
    dmean_acc = 100. * dmean_correct / total
    print("Epoch: %d | Test Acc: Res: %.2f%%  dmean: %.2f%%" %(epoch, res_acc, dmean_acc))  
    return res_acc

def get_discrepancy(output1, output2, noisy_label, class_balance=False, prior_class_weight=None):
    output1 = F.softmax(output1, dim=-1)
    output2 = F.softmax(output2, dim=-1)

    if class_balance:
        if noisy_label is None:
            output_mean = (output1 + output2) / 2
            class_weight = output_mean.sum(0) / output_mean.sum()
            _, predicted_class = torch.max(output_mean, 1)
            discrepancy_weight = class_weight[predicted_class]
            discrepancy_weight = discrepancy_weight.detach()
        else:
            class_weight = prior_class_weight.cuda()
            discrepancy_weight = class_weight[noisy_label]
            discrepancy_weight = discrepancy_weight.detach()
            discrepancy_weight = discrepancy_weight / discrepancy_weight.sum()

    discrepancy = torch.abs(output1 - output2)
    discrepancy = discrepancy.sum(-1)
    if class_balance:
        discrepancy = (torch.abs(output1 - output2).sum(-1) * discrepancy_weight)
    return discrepancy


def eval_train_clustering(model,all_loss):
    model.eval()
    num_samples = 50000
    all_targets = torch.zeros(50000)
    normalized_features = torch.zeros((num_samples, args.dr_dim))
    predictions = torch.zeros(num_samples)
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in tqdm(enumerate(eval_loader), total=len(eval_loader), ncols=0):
            inputs, targets = inputs.cuda(), targets.cuda() 
            _, _, feature, _, _, outputs = model(inputs)
            normalized_fea = F.normalize(feature, dim=1)
            loss = CE(outputs, targets)  
            _, predicted = torch.max(outputs, 1)

            for b in range(inputs.size(0)):
                all_targets[index[b]] = targets[b]
                normalized_features[index[b]] = normalized_fea[b]
                predictions[index[b]] = predicted[b]

    overall_distance = torch.zeros((num_samples, ))
    all_prob = np.zeros((num_samples,))
    centers = torch.zeros((args.num_class, args.dr_dim))
    for cls_ in range(args.num_class):
        centers[cls_] = F.normalize(normalized_features[predictions==cls_].mean(dim=0), dim=0)

    for cls_ in range(args.num_class):
        distance_cls = (normalized_features[all_targets==cls_] * centers[cls_]).sum(dim=1)
        overall_distance[all_targets==cls_] = distance_cls
    overall_distance = (overall_distance-overall_distance.min())/(overall_distance.max()-overall_distance.min())    
    overall_distance = overall_distance.numpy().reshape(-1, 1)
    gmm = GaussianMixture(n_components=2,max_iter=100,tol=1e-1,reg_covar=5e-4)
    gmm.fit(overall_distance)

    prob = gmm.predict_proba(overall_distance) 
    prob = prob[:,gmm.means_.argmax()]    
    all_prob = prob
    return all_prob, all_loss


def eval_train_discrepancy(epoch, model, iteration, discrepancy_loader, discrepancy_optimizer):
    model.train()

    discrepancy_loader_iter = iter(discrepancy_loader)
    prior_class_weight = torch.zeros((args.num_class,))
    for dataitem in discrepancy_loader:
        _, _, targets, _, _ = dataitem
        for cls_ in range(args.num_class):
            prior_class_weight[cls_] += ((targets == cls_).sum())
    prior_class_weight /= (prior_class_weight.sum())

    pbar = tqdm(range(iteration), total=iteration, ncols=0)
    for batch_idx in pbar:
        model.train()
        try:
            data_item = discrepancy_loader_iter.next()
        except:
            discrepancy_loader_iter = iter(discrepancy_loader)
            data_item = discrepancy_loader_iter.next()

        inputs, inputs_2, targets, _, _ = data_item
        inputs, targets = inputs.cuda(), targets.cuda()

        _, _, _, output1, output2, _ = model(inputs, detach_feature=True)
        discrepancy = get_discrepancy(output1, output2, targets, class_balance=True, prior_class_weight=prior_class_weight)

        loss = - discrepancy.mean() * args.max_discrepancy_weight
        discrepancy_optimizer.zero_grad()
        loss.backward()
        discrepancy_optimizer.step()

        pbar.set_description('Discrepancy: | Iter[%3d/%3d]\t Discre Loss: %.4f'
            %(batch_idx, iteration, discrepancy.mean().item()))

    prob = get_prob_paths(model, discrepancy_loader, info=False)
    return prob


def get_prob_paths(model, eval_loader, info=False):
    model.eval()
    num_samples = 50000
    discrepancies = torch.ones(num_samples)
    index_taken = torch.zeros(num_samples)
    all_targets = torch.zeros(num_samples)

    with torch.no_grad():
        pbar = tqdm(enumerate(eval_loader), total=len(eval_loader), ncols=0)
        for batch_idx, dataitem in pbar:
            inputs, _, targets, prob, index = dataitem
            inputs, targets = inputs.cuda(), targets.cuda() 
            _, _, _, output1, output2, outputs = model(inputs)
            discrepancy = get_discrepancy(output1, output2, targets)
            for b in range(inputs.size(0)):
                discrepancies[index[b]] = discrepancy[b]
                index_taken[index[b]] = 1
                all_targets[index[b]] = targets[b]

    index_taken = index_taken.int().numpy()
    assert not (index_taken>1).all()
    assert not (index_taken<0).all()
    assert (index_taken==0).sum() + (index_taken==1).sum() == 50000
    discrepancies_select = discrepancies[index_taken==1]
    targets_select = all_targets[index_taken==1]
    selected_sample_num = discrepancies_select.shape[0]
    prob = np.zeros((selected_sample_num,))
    occupied = np.zeros((selected_sample_num, ))
    for cls_ in range(args.num_class):
        discrepancies_cls = discrepancies_select[targets_select==cls_]
        assert discrepancies_cls.shape[0] > 1
        prob_cls = gmm_function(discrepancies_cls)
        prob[(targets_select==cls_).cpu().numpy()] = prob_cls
        occupied[(targets_select==cls_).cpu().numpy()] = 1
    assert occupied.all()
    prob_all = -np.ones((50000, ))
    prob_all[np.where(index_taken)] = prob

    return prob_all

def gmm_function(probs):
    probs = (probs-probs.min())/(probs.max()-probs.min())    
    probs = probs.reshape(-1,1)
    gmm = GaussianMixture(n_components=2,max_iter=10,reg_covar=5e-4,tol=1e-2)
    gmm.fit(probs)
    prob = gmm.predict_proba(probs) 
    prob = prob[:,gmm.means_.argmin()]       
    return prob

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

stats_log=open('./checkpoint/%s_%.1f_%s'%(args.dataset,args.idn_noise,args.noise_mode)+'_stats.txt','w') 
test_log=open('./checkpoint/%s_%.1f_%s'%(args.dataset,args.idn_noise,args.noise_mode)+'_acc.txt','w')     

if args.warmup_epoch != 0:
    warm_up = args.warmup_epoch
else:
    warm_up = 15

loader = dataloader.cifar_dataloader(args.dataset,batch_size=args.batch_size,num_workers=5,\
    root_dir=args.data_path,log=stats_log,noise_file='%s/%.1f_%s.json'%(args.data_path,args.idn_noise,args.noise_mode), idn_noise=args.idn_noise)

print('| Building net')
net1 = create_model_1fc()
net2 = create_model_1fc()
cudnn.benchmark = True

criterion = SemiLoss()
optimizer_res1 = optim.SGD(list(net1.model_resnet.parameters()) + list(net1.classification_fc.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer_res2 = optim.SGD(list(net2.model_resnet.parameters()) + list(net2.classification_fc.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)

optimizer_other1 = optim.SGD(list(net1.dr.parameters()) + list(net1.fc1.parameters()) + list(net1.fc2.parameters()), lr=args.lr_other, momentum=0.9, weight_decay=5e-4)
optimizer_other2 = optim.SGD(list(net2.dr.parameters()) + list(net2.fc1.parameters()) + list(net2.fc2.parameters()), lr=args.lr_other, momentum=0.9, weight_decay=5e-4)

optimizer_discrepancy1 = optim.SGD(list(net1.fc1.parameters()) + list(net1.fc2.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer_discrepancy2 = optim.SGD(list(net2.fc1.parameters()) + list(net2.fc2.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
if args.noise_mode=='asym':
    conf_penalty = NegEntropy()

all_loss = [[],[]] # save the history of losses from two networks

for epoch in range(args.num_epochs+1):   
    lr=args.lr
    if epoch >= 150:
        lr /= 10      
    for param_group in optimizer_res1.param_groups: param_group['lr'] = lr     
    for param_group in optimizer_res2.param_groups: param_group['lr'] = lr    
    for param_group in optimizer_other1.param_groups: param_group['lr'] = lr 
    for param_group in optimizer_other2.param_groups: param_group['lr'] = lr 
    for param_group in optimizer_discrepancy1.param_groups: param_group['lr'] = lr 
    for param_group in optimizer_discrepancy2.param_groups: param_group['lr'] = lr 
    
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')   
    
    if epoch<warm_up:       
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(epoch,net1,optimizer_res1,warmup_trainloader,optimizer_other1)    
        print('\nWarmup Net2')
        warmup(epoch,net2,optimizer_res2,warmup_trainloader,optimizer_other2) 
    else:         
        prob1,all_loss[0]=eval_train_clustering(net1,all_loss[0])   
        prob2,all_loss[1]=eval_train_clustering(net2,all_loss[1])          
            
        pred1 = (prob1 > args.p_threshold)      
        pred2 = (prob2 > args.p_threshold)      
        
        discrepancy_iter = args.dis_iter
        
        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('second_eval',pred2,prob2) # co-divide
        d_prob2 = eval_train_discrepancy(epoch, net2, discrepancy_iter, labeled_trainloader, optimizer_discrepancy2)
        d_pred2 = (d_prob2 >= args.d_threshold)
        labeled_trainloader, unlabeled_trainloader = loader.run('train',d_pred2,prob2) # co-divide
        train(epoch,net1,net2,optimizer_res1,labeled_trainloader, unlabeled_trainloader, optimizer_other1) # train net1  
        
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('second_eval',pred1,prob1) # co-divide
        d_prob1 = eval_train_discrepancy(epoch, net1, discrepancy_iter, labeled_trainloader, optimizer_discrepancy1)
        d_pred1 = (d_prob1 >= args.d_threshold)
        labeled_trainloader, unlabeled_trainloader = loader.run('train',d_pred1,prob1) # co-divide
        train(epoch,net2,net1,optimizer_res2,labeled_trainloader, unlabeled_trainloader, optimizer_other2) # train net2         

    test_epoch(net1,test_loader)  
    test_epoch(net2,test_loader)  
    test_net_mean(epoch, net1, net2)