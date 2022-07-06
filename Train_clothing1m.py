from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
import dataloader_clothing1M as dataloader
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from skimage.filters import threshold_otsu
from models.networks import create_model_discrepancy
from utils import parser_bool

parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.002, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=80, type=int)
parser.add_argument('--id', default='clothing1m')
parser.add_argument('--data_path', default='../dataset/clothing1m/', type=str, help='path to dataset')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=14, type=int)
parser.add_argument('--num_batches', default=1000, type=int)
parser.add_argument('--loss_type', default='l1', choices=['kl', 'l1'])
parser.add_argument('--dis_iter', default=1500, type=int)
parser.add_argument('--temperature', default=0.07, type=float, help='softmax temperature (default: 0.07)')
parser.add_argument('--gpu_num', default=1, type=int)
parser_bool(parser, 'other_lx', default=True)
parser_bool(parser, 'other_lx_penalty', default=True)
parser.add_argument('--min_discrepancy_weight', default=0.1, type=float)
parser.add_argument('--max_discrepancy_weight', default=2.3, type=float)
parser.add_argument('--num_workers', default=5, type=int)
parser.add_argument('--loss_output', default='other', choices=['other', 'res'])
parser.add_argument('--num_batches_eval', default=5000, type=int)

args = parser.parse_args()
args.id = __file__.split('/')[-1]

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

# Training
def train(epoch,net,net2,optimizer_res,labeled_trainloader,unlabeled_trainloader, optimizer_other):
    net.train()
    net2.eval()
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    pbar = tqdm(enumerate(labeled_trainloader), total=len(labeled_trainloader), ncols=0)
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in pbar:
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
        
        mixed_input = l * input_a[:batch_size*2] + (1 - l) * input_b[:batch_size*2]        
        mixed_target = l * target_a[:batch_size*2] + (1 - l) * target_b[:batch_size*2]
                
        _, logits, _, output_d1, output_d2, _ = net(mixed_input, detach_feature=True)

        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        # Lx on other
        Lx_other = -torch.mean(torch.sum(F.log_softmax(output_d1, dim=1) * mixed_target, dim=1)) + \
            -torch.mean(torch.sum(F.log_softmax(output_d2, dim=1) * mixed_target, dim=1))
        Lx_other = Lx_other / 2
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean_d1 = torch.softmax(output_d1, dim=1).mean(0)
        pred_mean_d2 = torch.softmax(output_d2, dim=1).mean(0)
        penalty_other = torch.sum(prior*torch.log(prior/pred_mean_d1)) + torch.sum(prior*torch.log(prior/pred_mean_d2))
        penalty_other = penalty_other / 2
        loss_other = Lx_other + penalty_other

        loss = Lx + penalty
        optimizer_res.zero_grad()
        loss.backward()
        optimizer_res.step()
        optimizer_other.zero_grad()
        loss_other.backward()
        optimizer_other.step()

        _, _, _, output_d1, output_d2, _ = net(mixed_input)
        L_discrepancy_other = get_discrepancy(output_d1, output_d2, mixed_target, loss_type=args.loss_type)
        loss_dis = L_discrepancy_other.mean() * args.min_discrepancy_weight

        optimizer_res.zero_grad()
        loss_dis.backward()
        optimizer_res.step()  

        pbar.set_description('Clothing1M | Epoch [%3d/%3d] Iter[%3d/%3d]\t  Labeled loss: %.4f Penalty: %.4f Contra Loss: %.4f Lx_other: %.4f Penalty_other: %.4f, loss_dis: %.4f'
                %(epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), penalty.item(), 0, Lx_other.item(), penalty_other.item(), loss_dis.item()))


def warmup(net,optimizer_res,dataloader,optimizer_other):
    net.train()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), ncols=0)
    for batch_idx, (inputs, labels, path) in pbar:
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer_res.zero_grad()
        _, res_out, _, output1, output2, _ = net(inputs)
        loss = CEloss(res_out, labels)
        penalty = (conf_penalty(res_out))
        L = loss + penalty       
        L.backward()  
        optimizer_res.step() 

        optimizer_other.zero_grad()
        _, res_out, _, output1, output2, _ = net(inputs)
        loss = CEloss(output1, labels) + CEloss(output2, labels)
        loss /= 2
        penalty = (conf_penalty(output1) + conf_penalty(output2)) / 2
        L = loss + penalty       
        L.backward()  
        optimizer_other.step() 

        pbar.set_description('|Warm-up: Iter[%3d/%3d]\t CE-loss: %.4f  Conf-Penalty: %.4f'
                %(batch_idx+1, args.num_batches, loss.item(), penalty.item()))


def val(net,val_loader,k):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, outputs, _, _, _, _ = net(inputs)
            _, predicted = torch.max(outputs, 1)         
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()              
    acc = 100.*correct/total
    print("\n| Validation\t Net%d  Acc: %.2f%%" %(k,acc))  
    if acc > best_acc[k-1]:
        best_acc[k-1] = acc
        print('| Saving Best Net%d ...'%k)
        save_point = './checkpoint/%s_net%d.pth.tar'%(args.id,k)
        torch.save(net.state_dict(), save_point)
    return acc

def get_discrepancy(output1, output2, noisy_label, loss_type='l1', class_balance=False, prior_class_weight=None):
    output1, output2 = F.softmax(output1, dim=-1), F.softmax(output2, dim=-1)
    if class_balance:
        class_weight = prior_class_weight.cuda()
        discrepancy_weight = class_weight[noisy_label]
        discrepancy_weight = discrepancy_weight.detach()
        discrepancy_weight = discrepancy_weight / discrepancy_weight.sum()
    assert loss_type == 'l1'
    discrepancy = torch.abs(output1 - output2).sum(-1)
    if class_balance: discrepancy = discrepancy * discrepancy_weight
    return discrepancy    

def eval_train_clustering(epoch, model, eval_loader):
    model.eval()
    num_samples = len(eval_loader.dataset)
    losses = torch.zeros(num_samples).cuda()
    all_targets = torch.zeros(num_samples).cuda()
    paths = []

    normalized_features = torch.zeros((num_samples, 128))
    predictions = torch.zeros(num_samples)
    n=0
    with torch.no_grad():
        for _, (inputs, targets, path) in tqdm(enumerate(eval_loader), total=len(eval_loader), ncols=0):
            inputs, targets = inputs.cuda(), targets.cuda() 
            _, _, feature, _, _, outputs = model(inputs)
            normalized_fea = F.normalize(feature, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_targets[n: n+inputs.size(0)] = targets
            paths += path
            normalized_features[n:n+inputs.size(0)] = normalized_fea
            predictions[n:n+inputs.size(0)] = predicted
            n += inputs.size(0)
    all_targets = all_targets.cpu()
    assert len(paths) == num_samples

    all_prob = np.zeros((num_samples,))
    centers = torch.zeros((14, 128))
    for cls_ in range(14):
        centers[cls_] = F.normalize(normalized_features[predictions==cls_].mean(dim=0), dim=0)
    for cls_ in range(14):
        distance_cls = (normalized_features[all_targets==cls_] * centers[cls_]).sum(dim=1)
        threshold = threshold_otsu(distance_cls.reshape(-1, 1).numpy())
        prob_cls = (distance_cls > threshold).float()
        all_prob[all_targets==cls_] = prob_cls
    return all_prob, paths

def eval_train_discrepancy(epoch, model, iteration, discrepancy_loader, discrepancy_optimizer):
    model.train()
    discrepancy_loader_iter = iter(discrepancy_loader)
    prior_class_weight = torch.zeros((14,))
    for dataitem in discrepancy_loader:
        inputs, targets, _ = dataitem
        for cls_ in range(14):
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

        if len(data_item) == 2: inputs, targets = data_item
        elif len(data_item) == 3: inputs, targets, _ = data_item

        inputs, targets = inputs.cuda(), targets.cuda()
        _, _, _, output1, output2, _ = model(inputs, detach_feature=True)
        discrepancy = get_discrepancy(output1, output2, targets, class_balance=True, loss_type=args.loss_type, prior_class_weight=prior_class_weight)
        loss = - discrepancy.mean() * args.max_discrepancy_weight
        discrepancy_optimizer.zero_grad()
        loss.backward()
        discrepancy_optimizer.step()

        pbar.set_description('Discrepancy: | Iter[%3d/%3d]\t Discre Loss: %.4f'
            %(batch_idx, iteration, discrepancy.mean().item()))

    prob, paths = get_prob_paths(model, discrepancy_loader, info=False)
    return prob, paths
    

def get_prob_paths(model, eval_loader, info=False):
    model.eval()
    num_samples = len(eval_loader.dataset)
    losses = torch.zeros(num_samples)
    discrepancies = torch.zeros(num_samples)
    predictions = torch.zeros(num_samples).long()
    all_targets = torch.zeros(num_samples).long()
    paths = []
    n=0
    with torch.no_grad():
        pbar = tqdm(enumerate(eval_loader), total=len(eval_loader), ncols=0)
        for batch_idx, dataitem in pbar:
            inputs, targets, path = dataitem
            inputs, targets = inputs.cuda(), targets.cuda() 
            _, _, _, output1, output2, outputs = model(inputs)
            discrepancy = get_discrepancy(output1, output2, targets, loss_type=args.loss_type)
            loss = CE(outputs, targets)  
            _, pred = torch.max(outputs, 1)         
            for b in range(inputs.size(0)):
                losses[n]=loss[b] 
                discrepancies[n] = discrepancy[b]
                paths.append(path[b])
                predictions[n] = pred[b]
                all_targets[n] = targets[b]
                n+=1
    for cls_ in range(14):
        assert cls_ in all_targets.unique()
        assert (all_targets == cls_).sum() >= 2
    
    prob2 = np.zeros((num_samples,))
    occupied = np.zeros((num_samples, ))
    for cls_ in range(14):
        discrepancies_cls = discrepancies[all_targets==cls_]
        assert discrepancies_cls.shape[0] > 1
        prob2_cls = gmm_function(discrepancies_cls)
        prob2[(all_targets==cls_).cpu().numpy()] = prob2_cls
        occupied[(all_targets==cls_).cpu().numpy()] = 1
    assert occupied.all()
    return prob2, paths

def gmm_function(samples):
    samples = (samples - samples.min())/(samples.max() - samples.min())    
    samples = samples.reshape(-1,1)
    gmm = GaussianMixture(n_components=2,max_iter=10,reg_covar=5e-4,tol=1e-2)
    gmm.fit(samples)
    prob = gmm.predict_proba(samples) 
    prob = prob[:,gmm.means_.argmin()]       
    return prob   

def test(net1, net2, test_loader):
    net1.eval()
    net2.eval()
    res_correct = 0
    d1_correct = 0
    d2_correct = 0
    dmean_correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader), ncols=0)
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.cuda(), targets.cuda()
            _, res_out_a, _, _, _, output_mean_a = net1(inputs)
            _, res_out_b, _, _, _, output_mean_b = net2(inputs)
            res_out = res_out_a + res_out_b
            output_mean = output_mean_a + output_mean_b

            _, res_predicted = torch.max(res_out, 1)
            _, dmean_predicted = torch.max(output_mean, 1)
            total += targets.size(0)
            res_correct += res_predicted.eq(targets).cpu().sum().item()
            dmean_correct += dmean_predicted.eq(targets).cpu().sum().item()
    res_acc = 100. * res_correct / total
    dmean_acc = 100. * dmean_correct / total
    print("\n| 2 Net Test Acc: Res: %.2f%%  dmean: %.2f%%\n" %(res_acc, dmean_acc))  
    return res_acc

print("| Building loader")
loader = dataloader.clothing_dataloader(root=args.data_path,batch_size=args.batch_size,\
    num_workers=args.num_workers,num_batches=args.num_batches)

print('| Building net')
net1 = create_model_discrepancy(args)
net2 = create_model_discrepancy(args)
cudnn.benchmark = True

optimizer_res1 = optim.SGD(list(net1.model_resnet.parameters()) + list(net1.classification_fc.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-3)
optimizer_res2 = optim.SGD(list(net2.model_resnet.parameters()) + list(net2.classification_fc.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-3)
optimizer_other1 = optim.SGD(list(net1.dr.parameters()) + list(net1.fc1.parameters()) + list(net1.fc2.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-3)
optimizer_other2 = optim.SGD(list(net2.dr.parameters()) + list(net2.fc1.parameters()) + list(net2.fc2.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-3)
optimizer_discrepancy1 = optim.SGD(list(net1.fc1.parameters()) + list(net1.fc2.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-3)
optimizer_discrepancy2 = optim.SGD(list(net2.fc1.parameters()) + list(net2.fc2.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-3)
                    
CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()

best_acc = [0,0]
for epoch in range(args.num_epochs+1):   
    lr=args.lr
    if epoch >= 40:
        lr /= 10       
    for param_group in optimizer_res1.param_groups: param_group['lr'] = lr     
    for param_group in optimizer_res2.param_groups: param_group['lr'] = lr    
    for param_group in optimizer_other1.param_groups: param_group['lr'] = lr 
    for param_group in optimizer_other2.param_groups: param_group['lr'] = lr 
    for param_group in optimizer_discrepancy1.param_groups: param_group['lr'] = lr 
    for param_group in optimizer_discrepancy2.param_groups: param_group['lr'] = lr 
        
    if epoch<1:
        train_loader = loader.run('warmup', warmup_batches_num=1000)
        print('Warmup Net1')
        warmup(net1,optimizer_res1,train_loader, optimizer_other1)             
        train_loader = loader.run('warmup', warmup_batches_num=1000)
        print('\nWarmup Net2')
        warmup(net2,optimizer_res2,train_loader, optimizer_other2)                  
    else:       
        pred1 = (prob1 > args.p_threshold)  # divide dataset  
        pred2 = (prob2 > args.p_threshold)      
        print('\n\nTrain Net1')
        labeled_trainloader_1, unlabeled_trainloader_1 = loader.run('train',pred2,prob2,paths=paths2, eval_selection=True) # co-divide
        train(epoch, net1, net2, optimizer_res1, labeled_trainloader_1, unlabeled_trainloader_1, optimizer_other1)
        print('\n\nTrain Net2')
        labeled_trainloader_2, unlabeled_trainloader_2 = loader.run('train',pred1,prob1,paths=paths1, eval_selection=True) # co-divide
        train(epoch, net2, net1, optimizer_res2, labeled_trainloader_2, unlabeled_trainloader_2, optimizer_other2)

    # validation
    if epoch>=1:
        val_loader = loader.run('val') # validation
        acc1 = val(net1,val_loader,1)
        acc2 = val(net2,val_loader,2)
    # test
    test_loader = loader.run('test')
    acc = test(net1, net2, test_loader)

    print("Filtering For Net 1......")
    eval_loader = loader.run('eval_train', num_batches_eval=args.num_batches_eval, gpu_num=args.gpu_num)
    prob1, paths1 = eval_train_clustering(epoch, net1, eval_loader)
    pred1 = (prob1 > args.p_threshold)
    divided_loader = loader.run('train_eval',pred1,prob1,paths=paths1, gpu_num=args.gpu_num)
    prob1,paths1 = eval_train_discrepancy(epoch,net1,args.dis_iter,divided_loader[0],optimizer_discrepancy1)
    pred1 = (prob1 > args.p_threshold)  # divide dataset  
    _, prob1, paths1 = eval_loader.dataset.selection_by_cls(pred1, prob1, paths1, 32000)

    print("Filtering For Net 2......")
    eval_loader = loader.run('eval_train', num_batches_eval=args.num_batches_eval, gpu_num=args.gpu_num)
    prob2, paths2 = eval_train_clustering(epoch, net2, eval_loader)
    pred2 = (prob2 > args.p_threshold)
    divided_loader = loader.run('train_eval',pred2,prob2,paths=paths2, gpu_num=args.gpu_num)
    prob2,paths2 = eval_train_discrepancy(epoch,net2,args.dis_iter,divided_loader[0],optimizer_discrepancy2)
    pred2 = (prob2 > args.p_threshold)
    _, prob2, paths2 = eval_loader.dataset.selection_by_cls(pred2, prob2, paths2, 32000)

test_loader = loader.run('test')
net1.load_state_dict(torch.load('./checkpoint/%s_net1.pth.tar'%args.id))
net2.load_state_dict(torch.load('./checkpoint/%s_net2.pth.tar'%args.id))
acc = test(net1,net2,test_loader)      
