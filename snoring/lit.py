# %%
import lightning as pl
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2
import torch
import torch.nn as nn  
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score
from torchmetrics.functional import accuracy, precision, recall, f1_score
average = 'macro'

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    F1_score = f1_score(pred_flat, labels_flat, average='macro')

    return accuracy_score(pred_flat, labels_flat), F1_score

class TFModule(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.f_module =  nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 3), stride=(1, 2), bias=False),
            nn.Conv2d(8, 16, kernel_size=(1, 3), stride=(1, 2), bias=False),
            nn.Conv2d(16, 32, kernel_size=(1, 3), stride=(1, 2), bias=False),
            nn.Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 2), bias=False),
            nn.AdaptiveMaxPool2d((128, 1)),
            nn.Conv2d(32, 1, kernel_size=(1, 1), bias=False),
            # nn.Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 2), bias=False),
            # nn.Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 2), bias=False)
        )
        
        self.t_module =  nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(5, 1), stride=(2, 1), bias=False),
            nn.Conv2d(8, 16, kernel_size=(5, 1), stride=(2, 1), bias=False),
            nn.Conv2d(16, 32, kernel_size=(5, 1), stride=(2, 1), bias=False),
            nn.Conv2d(32, 32, kernel_size=(5, 1), stride=(2, 1), bias=False),
            nn.AdaptiveMaxPool2d((1, 345)),
            nn.Conv2d(32, 1, kernel_size=(1, 1), bias=False),
            # nn.Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 2), bias=False),
            # nn.Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 2), bias=False)
        )
    
    def forward(self, x):
        
        x_har = x[:,0:1]
        x_pess = x[:,1:2]
        
        # print(x_har.shape)
        print("x_har", x_har.shape)
        print("x_pess", x_pess.shape)
        
        f_score = self.f_module(x_har)
        print("f_score", f_score.shape)
        
        # print("f_score", f_score.shape)
        
        t_score = self.t_module(x_pess)
        print("t_score", t_score.shape)
        x_har = x_har * f_score 
        x_pess = x_pess * t_score
        
        # print(x_har.shape)
        # x = x*torch.stack(f_score, t_score)
        
        
        
                
        return torch.concat((x_har, x_pess),1)

class LitClassification(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.model = mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 10)
        
        
        self.model.features[0][0] = nn.Conv2d(2, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        self.tf_module = TFModule()
        
        self.all_preds = []
        self.all_labels = []
    
    def forward(self, x):
        # print()
        x = x[:,1:]
        # print(x.shape)
        x = self.tf_module(x)

        # print(x.shape)
        x = self.model(x)
        
        return x
        
        
        # forward
    

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)

        return optimizer
    
    def share_batch(self, batch, state):

        inputs, targets = batch['data'], batch['label']
        
        outputs = self(inputs)
        
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        
        if state == "Train":
            pass
            
            # acc_batch, f1_batch = flat_accuracy(outputs.cpu(), targets.cpu())
        
            # acc = self.acc(student_logits, labels)
            # self.log(f'{state}_acc', acc_batch, on_step=False, on_epoch=True, prog_bar=True)
        elif state == "valid":
            pred = outputs.argmax(dim=1)
            self.all_preds.append(pred.to('cpu'))
            self.all_labels.append(targets.to('cpu'))

        self.log(f"{state}_loss", loss, on_step=False, on_epoch=True)
#         self.log(f"{state}_rep_loss", rep_loss, on_step=False, on_epoch=True)
#         self.log(f"{state}_att_loss", att_loss, on_step=False, on_epoch=True)
#         self.log(f"{state}_cls_loss", cls_loss, on_step=False, on_epoch=True)

        return loss
    
    def on_validation_epoch_end(self):

        all_preds = torch.cat(self.all_preds,dim=0)
        all_labels = torch.cat(self.all_labels,dim=0)
        # print(all_preds.shape)
        acc = accuracy(all_preds, all_labels, task="multiclass", num_classes=10)
        pre = precision(all_preds, all_labels, task="multiclass", average=average, num_classes=10)
        rec = recall(all_preds, all_labels, task="multiclass", average=average, num_classes=10)
        f1 = f1_score(all_preds, all_labels, task="multiclass", average=average, num_classes=10)
        
        self.log('val_acc', acc,  prog_bar=True)
        self.log('val_pre', pre)
        self.log('val_rec', rec)
        self.log('val_f1', f1)
        
        self.all_preds = []
        self.all_labels = []
    
    def training_step(self, train_batch, batch_idx):
        loss = self.share_batch(train_batch, "train")
        # print(loss)
        return loss

    def validation_step(self, val_batch, batch_idx):

        loss = self.share_batch(val_batch, "valid")

    def test_step(self, test_batch, batch_idx):

        loss = self.share_batch(test_batch, "test")
        
        
# %%
if __name__ == "__main__":
    x = torch.ones([5,3, 128, 345]) #87
    litmodel = LitClassification()  #TFModule()
    
    
    y = litmodel(x)
    
    print(y.shape)
    # print(y[0].shape, y[1].shape)
    
# %%
