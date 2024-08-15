import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

import dgl
from dgl.dataloading import GraphDataLoader

from baseline_model.custom_dataset import ProteinGraphDGLDataset, ProteinGraphDataset
from baseline_model.graph_transformer import GraphTransformerNet
from baseline_model.att_CNN import SelfAttention, CrossAttention, MLPLayer
from baseline_model.my_metrics import get_metrics


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available. Using GPU.')
else:
    device = torch.device('cpu')
    print('CUDA is not available. Using CPU.')


class CoAttentionModel(nn.Module):
    def __init__(self, node_num, num_features, plm_num, hidden_size, num_heads, dropout_prob, dropout_ratio=0.4) -> None:
        super().__init__()
        self.node_num = node_num  
        self.num_features = num_features 
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.plm_num = plm_num

        self.SA = SelfAttention(hidden_size, num_heads, dropout_prob)
        self.Co = CrossAttention(hidden_size, num_heads, dropout_prob)        
        self.plm_head = nn.Linear(1280, hidden_size)
        self.plm_layer_norm = nn.LayerNorm(hidden_size) 
        self.graphModel = GraphTransformerNet(net_params)
        self.graph_head = nn.Linear(num_features, hidden_size)
        self.attention = CrossAttention(hidden_size, num_heads, dropout_prob)
        self.batch_norm = nn.BatchNorm1d(hidden_size) 
        self.dropout = nn.Dropout(dropout_ratio)
        self.init_weights()
    
        #self.fc = MLPLayer(input_dim=node_num*hidden_size, output_dim=2, L=2)
        self.fc = nn.Sequential(
            nn.Linear(25*128*4, 128*4), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128*4, 128), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )        


    def init_weights(self):
        nn.init.xavier_uniform_(self.plm_head.weight)
        nn.init.constant_(self.plm_head.bias, 0)
        
        
    def forward(self, batched_graphs):
        plm_features = batched_graphs.ndata['plm_features']
        plm_output = torch.stack([plm_features[i:i + self.node_num] for i in range(0, len(plm_features), self.node_num)])
        plm_output = self.plm_head(plm_output)
        plm_output = self.plm_layer_norm(plm_output)
        

        node_features = batched_graphs.ndata['feat']
        graph_output = self.graphModel(batched_graphs, node_features)
        reshaped_graphoutput = torch.stack([graph_output[i:i + self.node_num] for i in range(0, len(graph_output), self.node_num)])
        reshaped_graphoutput = self.graph_head(reshaped_graphoutput)
        #reshaped_graphoutput = reshaped_graphoutput.reshape(batch_size, -1)


        plm_output_sa = self.SA(plm_output) 
        reshaped_graphoutput_sa = self.SA(reshaped_graphoutput)
        fused_output1, attention_map1 = self.Co(plm_output, reshaped_graphoutput)
        fused_output2, attention_map2 = self.Co(reshaped_graphoutput_sa, plm_output_sa)
        
        fused_output1 = self.Co(plm_output_sa, reshaped_graphoutput_sa) 
        fused_output2 = self.Co(reshaped_graphoutput_sa, plm_output_sa) 
        fused_output = torch.cat([fused_output1, fused_output2, plm_output, reshaped_graphoutput], dim=2)
        #fused_output = torch.cat([plm_output, reshaped_graphoutput], dim=2)
        #fused_output = self.batch_norm((fused_output + plm_output).permute(0, 2, 1)).permute(0, 2, 1) 
        fused_output = self.dropout(fused_output)
        fused_output = fused_output.reshape(fused_output.size(0), -1)

        classification_output = self.fc(fused_output)

        return classification_output



train_dataset = ProteinGraphDGLDataset(root='/train_graph_dataset')
test_dataset = ProteinGraphDGLDataset(root='/test_graph_dataset')
batch_size = 64
train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.1, random_state=4)

train_loader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = GraphDataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)


net_params = {
    'num_atom_type': 78, 
    'hidden_dim': 78,
    'n_heads': 1,
    'out_dim': 78,
    'in_feat_dropout': 0.0,
    'dropout': 0.0,
    'L':6,
    'layer_norm': True,
    'batch_norm': True,
    'residual': True,
    'lap_pos_enc': False,
    'wl_pos_enc': False  
}

node_num = 41  
num_features = 78  
plm_num = 1280 
hidden_size = 128 
num_heads = 1
dropout_prob = 0.3  
num_epochs = 30

model = CoAttentionModel(node_num, num_features, plm_num, hidden_size, num_heads, dropout_prob)
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1)


num_epochs = 30
best_val_loss = float('inf')
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batched_graphs, labels in train_loader:
        batched_graphs = batched_graphs.to(device) 
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(batched_graphs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}")

    model.eval()
    val_loss = 0
    for batched_graphs, labels in val_loader:
        batched_graphs = batched_graphs.to(device) 
        labels = labels.to(device)

        output = model(batched_graphs)
        loss = criterion(output, labels)
        val_loss += loss.item()
    val_loss /= len(val_loader)
    scheduler.step(val_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()

torch.save(best_model_state, 'best_model.pth')
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
all_labels = []
all_preds = []
probabilities = []

for batched_graphs, labels in test_loader:
    batched_graphs = batched_graphs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        output = model(batched_graphs)
        preds = output.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        softmax = nn.Softmax(dim=1)
        probs = softmax(output)
        probabilities.extend(probs[:, 1].cpu().numpy()) 

results = get_metrics(all_labels, all_preds, probabilities)
