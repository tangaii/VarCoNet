import torch
from torch import nn
from torch.optim import Adam
from model_scripts.evaluation import BaseEvaluator
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torch.nn import Linear
import copy



class MLP(nn.Module):
    def __init__(self, num_features, num_classes):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            Linear(num_features, num_classes),
            nn.Softmax(dim=1))

    def forward(self, x):
        z = self.fc(x)
        return z


class LREvaluator(BaseEvaluator):
    def __init__(self, num_epochs: int = 200, learning_rate: float = 0.0005, test_interval: int = 1):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.test_interval = test_interval
            

    def evaluate(self, encoder_model, X_train, y_train, X_val, y_val, X_test, y_test, num_classes, device):
        val_aucs = []
        val_losses = []
        test_aucs = []
        test_losses = []
        train_losses = []
        train_aucs = []
        best_val_auc = 0
        best_val_loss = 100000
        input_dim = X_train.shape[1]
        classifier = MLP(input_dim, num_classes).to(device)
        optimizer = Adam(classifier.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        
        for epoch in range(1,self.num_epochs):
            classifier.train()
            optimizer.zero_grad()
            output = classifier(X_train)
            class_loss = criterion(output, F.one_hot(y_train, num_classes=2).float())
            class_loss.backward()
            optimizer.step()
            if epoch % self.test_interval == 0:
                classifier.eval()
                with torch.no_grad():
                    output_train = classifier(X_train)
                    output_val = classifier(X_val)
                    output_test = classifier(X_test)
                    class_loss_train = class_loss.item()
                    class_loss_val = criterion(output_val, F.one_hot(y_val, num_classes=2).float()).item()
                    class_loss_test = criterion(output_test, F.one_hot(y_test, num_classes=2).float()).item()
                train_auc = roc_auc_score(y_train.cpu().numpy(), output_train[:,-1].cpu().numpy())
                val_auc = roc_auc_score(y_val.cpu().numpy(), output_val[:,-1].cpu().numpy())
                test_auc = roc_auc_score(y_test.cpu().numpy(), output_test[:,-1].cpu().numpy())
                val_aucs.append(val_auc)
                test_aucs.append(test_auc)
                val_losses.append(class_loss_val)
                test_losses.append(class_loss_test)
                train_losses.append(class_loss_train)
                train_aucs.append(train_auc)

                if class_loss_val < best_val_loss: 
                    best_train_auc = train_auc
                    best_val_auc = val_auc
                    best_test_auc = test_auc
                    best_train_loss = class_loss_train
                    best_val_loss = class_loss_val
                    best_test_loss = class_loss_test
                    best_train_probs = output_train[:,-1].cpu().numpy()
                    best_val_probs = output_val[:,-1].cpu().numpy()
                    best_test_probs = output_test[:,-1].cpu().numpy()
                    linear_state_dict = copy.deepcopy(classifier.state_dict())
                    
        print('')
        print(f'train_auc: {best_train_auc:.2f} | train_loss: {best_train_loss:.2f}')
        print(f'val_auc  : {best_val_auc:.2f} | val_loss  : {best_val_loss:.3f}')
        print(f'test_auc : {best_test_auc:.2f} | test_loss : {best_test_loss:.2f}')

        return {
            'best_train_auc': best_train_auc,
            'best_val_auc': best_val_auc,
            'best_test_auc': best_test_auc,
            'best_train_loss': best_train_loss,
            'best_val_loss': best_val_loss,
            'best_test_loss': best_test_loss,
            'train_probs': best_train_probs,
            'val_probs': best_val_probs,
            'test_probs': best_test_probs,
            'y_train': y_train.cpu().numpy(),
            'y_val': y_val.cpu().numpy(),
            'y_test': y_test.cpu().numpy(),
        }, linear_state_dict
    
