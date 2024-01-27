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
        
        # f_module = 
        pass

class LitClassification(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.model = mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 10)
        
        
        self.model.features[0][0] = nn.Conv2d(2, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        self.all_preds = []
        self.all_labels = []
    

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

        return optimizer
    
    def share_batch(self, batch, state):

        inputs, targets = batch['data'], batch['label']
        
        outputs = self.model(inputs[:,1:])
        
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
    x = torch.ones([1, 128, 87])[None]
    litmodel = LitClassification(None,None,None)
    
    
    y = litmodel.model.features(x)
    
    
    print(y.shape)
    
# %%
