import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.optim import optimizer
import torchvision
import torchvision.models as models

def create_model_discrepancy(args):
    class ResClassifier(nn.Module):
        def __init__(self, class_num=14):
            super(ResClassifier, self).__init__()
            self.fc1 = nn.Sequential(
                nn.Linear(128, 64),
                nn.BatchNorm1d(64, affine=True),
                nn.ReLU(inplace=True),
                nn.Dropout()
                )
            self.fc2 = nn.Sequential(
                nn.Linear(64, 64),
                nn.BatchNorm1d(64, affine=True),
                nn.ReLU(inplace=True),
                nn.Dropout()
                )
            self.fc3 = nn.Linear(64, class_num)
        def forward(self, x):
            fc1_emb = self.fc1(x)
            fc2_emb = self.fc2(fc1_emb)    
            logit = self.fc3(fc2_emb)
            return logit
    class MyModel(nn.Module):
        def __init__(self, num_classes1, num_classes2):
            super(MyModel, self).__init__()
            assert num_classes1 == num_classes2
            self.num_classes = num_classes1
            self.model_resnet = models.resnet50(pretrained=True)
            num_ftrs = self.model_resnet.fc.in_features
            self.model_resnet.fc = nn.Identity()
            self.classification_fc = nn.Linear(num_ftrs, num_classes1)
            self.dr = nn.Linear(num_ftrs, 128)
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
