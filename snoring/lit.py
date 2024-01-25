# %%
import lightning as pl
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2
import torch
import torch.nn as nn  
import torch.optim as optim


class LitClassification(pl.LightningModule):
    def __init__(self, data_module, df_train, df_test):
        super().__init__()
        
        self.data_module = data_module
        self.df_train = df_train
        self.df_test = df_test

        # config = Wav2Vec2Config.from_json_file("config.json")
        self.model = mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 2)
        
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # self.teacher_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base-960h", num_labels=10)
        # self.teacher_model.load_state_dict(state_dict_teacher_new)
        # self.student_model = Wav2Vec2ForSequenceClassification(config)

#         self.fit_dense = nn.Linear(config.hidden_size, 768)
#         self.project_logits = nn.Sequential(
#             nn.Linear(config.hidden_size, config.hidden_size),
#             nn.ReLU(),
#             nn.Linear(config.hidden_size, config.hidden_size),
#             nn.ReLU(),
#             nn.Linear(config.hidden_size,(config.num_hidden_layers+1)*13)
# #             nn.Softmax(-1)
#         )
        
        # self.acc = torchmetrics.Accuracy("multiclass", num_classes=10)
        # self.f1 = torchmetrics.F1Score("multiclass",num_classes=10, average='macro')
        
#         self.teacher_model.eval()
#         print(self.teacher_model.training)
        self.all_preds = []
        self.all_labels = []
    
    
    def train_dataloader(self):
        datacry = self.data_module(self.df_train)
        return DataLoader(datacry, batch_size=32, num_workers=2)
    
    def val_dataloader(self):
        datacry = self.data_module(self.df_test)
        return DataLoader(datacry, batch_size=16, num_workers=2)
    
    def test_dataloader(self):
        datacry = self.data_module(self.df_test)
        return DataLoader(datacry, batch_size=16, num_workers=2)
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        # optimizer = AdamW(list(self.student_model.parameters())+
        #                   list(self.fit_dense.parameters())+
        #                   list(self.project_logits.parameters()),
        #     lr = 1e-3, # args.learning_rate - default is 5e-5,
        #     eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
        # )

        # total_steps = len(self.train_dataloader()) * 200

        # # Create the learning rate scheduler.
        # scheduler = get_linear_schedule_with_warmup(optimizer, 
        #                 num_warmup_steps = 1000, # Default value in run_glue.py
        #                 num_training_steps = total_steps)
        # return [optimizer], [scheduler]
        return optimizer
    
    def share_batch(self, batch, state):
        inputs, targets = batch
        
        outputs = self.model(inputs)
        
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        
        
        
#         print(labels)
        
        # print(input_values.shape)
#         out_student = self.student_model(input_values, 
#                 attention_mask=attention_masks, 
#                 output_hidden_states=True,
#                 output_attentions=True,
#                 )
#         self.teacher_model.eval()
#         with torch.no_grad():
#             out_teacher = self.teacher_model(input_values, 
#                 attention_mask=attention_masks, 
#                 output_hidden_states=True,
#                 output_attentions=True,
#                 )

#         att_loss = 0.
#         rep_loss = 0.
#         cls_loss = 0.
#         akd_loss = 0.
        
#         student_logits, student_atts, student_reps = out_student.logits, out_student.attentions, out_student.hidden_states
#         teacher_logits, teacher_atts, teacher_reps = out_teacher.logits, out_teacher.attentions, out_teacher.hidden_states
#         # print(input_values.shape)
#         teacher_layer_num = len(teacher_atts)
#         student_layer_num = len(student_atts)
        
#         assert teacher_layer_num % student_layer_num == 0
#         layers_per_block = int(teacher_layer_num / student_layer_num)
#         new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
#                             for i in range(student_layer_num)]

#         for student_att, teacher_att in zip(student_atts, new_teacher_atts):
# #             print("teacher_att",teacher_att.shape)
#             tmp_loss = F.mse_loss(student_att, teacher_att)
#             att_loss += tmp_loss
        
#         hidden_states = student_reps[-1][:,0]
#         teacher_reps_concat = torch.stack(teacher_reps,1)
    
#         alphas = self.project_logits(hidden_states).chunk(student_layer_num+1, -1)

#         b,t,h = student_reps[0].shape
#         new_student_reps = student_reps
#         for i, (student_rep, alpha) in enumerate(zip(new_student_reps, alphas)):
#             student_rep = model_lit.fit_dense(student_rep)
#             alpha = alpha.softmax(dim=-1)
#             alpha = torch.reshape(alpha, (b, teacher_layer_num+1,1,1))
#             teacher_reps_att = teacher_reps_concat * alpha
#             teacher_reps_att = torch.sum(teacher_reps_att, 1)
#             tmp_loss = F.mse_loss(student_rep, teacher_reps_att)
#             akd_loss += tmp_loss
            
#         new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
#         new_student_reps = student_reps
#         for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
#             # print(student_rep.shape, teacher_rep.shape, )
#             tmp_loss = F.mse_loss(self.fit_dense(student_rep), teacher_rep)
#             rep_loss += tmp_loss

        
#         cls_loss = soft_cross_entropy(student_logits / 1.,
#                                 teacher_logits / 1.)
        
#         loss = akd_loss + cls_loss + att_loss
        
#         if state == "train":
#             acc = self.acc(student_logits, labels)
#             self.log(f'{state}_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
#         elif state == "valid":
#             pred = student_logits.argmax(dim=1)
#             self.all_preds.append(pred.to('cpu'))
#             self.all_labels.append(labels.to('cpu'))

#         self.log(f"{state}_loss", loss, on_step=False, on_epoch=True)
#         self.log(f"{state}_rep_loss", rep_loss, on_step=False, on_epoch=True)
#         self.log(f"{state}_att_loss", att_loss, on_step=False, on_epoch=True)
#         self.log(f"{state}_cls_loss", cls_loss, on_step=False, on_epoch=True)

        return loss
    
    def on_validation_epoch_end(self):

        all_preds = torch.cat(self.all_preds,dim=0)
        all_labels = torch.cat(self.all_labels,dim=0)
        # print(all_preds.shape)
        # acc = accuracy(all_preds, all_labels, task="multiclass", num_classes=10)
        # pre = precision(all_preds, all_labels, task="multiclass", average=average, num_classes=10)
        # rec = recall(all_preds, all_labels, task="multiclass", average=average, num_classes=10)
        # f1 = f1_score(all_preds, all_labels, task="multiclass", average=average, num_classes=10)
        
        # self.log('val_acc', acc,  prog_bar=True)
        # self.log('val_pre', pre)
        # self.log('val_rec', rec)
        # self.log('val_f1', f1)
        
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
