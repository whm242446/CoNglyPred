import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline_model.graph_model import GCNModel, GATModel


class SelfAttention(nn.Module):
    
    def __init__(self, hidden_size, num_heads, dropout_prob):   
        super(SelfAttention, self).__init__()
        if hidden_size % num_heads != 0: 
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_heads))
        
        self.num_heads = num_heads  
        self.attention_head_size = int(hidden_size / num_heads)
        self.all_head_size = int(self.num_heads * self.attention_head_size)   

        self.query = nn.Linear(hidden_size, self.all_head_size) 
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # dropout
        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):

        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size) 
        x = x.view(*new_x_shape)   
        return x.permute(0, 2, 1, 3)  


    def forward(self, graph_ouput, plm_output):
        mixed_query_layer = self.query(graph_ouput)    
        mixed_key_layer = self.key(plm_output)        
        mixed_value_layer = self.value(plm_output)      

        query_layer = self.transpose_for_scores(mixed_query_layer)   
        key_layer = self.transpose_for_scores(mixed_key_layer)       
        value_layer = self.transpose_for_scores(mixed_value_layer)   

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer) 
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() 
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)   
        attention_layer = context_layer.view(*new_context_layer_shape)
        
        return attention_layer   
    


# cross-attention
class CrossModelAttention(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, num_heads, dropout_prob, dropout_ratio=0.4) -> None:
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        
        self.plm_head = nn.Linear(1024, hidden_size)
        self.plm_layer_norm = nn.LayerNorm(hidden_size) 
        self.graphModel = GCNModel(num_features=num_features, hidden_size=hidden_size, num_layers=num_layers)
        self.attention = SelfAttention(hidden_size, num_heads, dropout_prob)
        self.batch_norm = nn.BatchNorm1d(hidden_size) 
        self.dropout = nn.Dropout(dropout_ratio)
        self.loss_fucntion = torch.nn.CrossEntropyLoss()
        self.init_weights()
        
        self.output_layer = nn.Linear(41*hidden_size, 2)

    def init_weights(self):
        nn.init.xavier_uniform_(self.plm_head.weight)
        nn.init.constant_(self.plm_head.bias, 0)
        
        
    def forward(self, batch):

        plm_features = torch.tensor(batch.plm_features).to(device)
        
        batch_size = plm_features.size(0)

        graph_features = batch.x
        edge_index = batch.edge_index
 
        plm_output = F.relu(self.plm_head(plm_features))
        plm_output = self.plm_layer_norm(plm_output) 

        graph_output = self.graphModel(graph_features, edge_index)
        n_features = 41
        reshaped_graphoutput = torch.stack([graph_output[i:i + n_features] for i in range(0, len(graph_output), n_features)])

        fused_output = self.attention(reshaped_graphoutput, plm_output)

        fused_output = self.batch_norm((fused_output + plm_output).permute(0, 2, 1)).permute(0, 2, 1) 
        fused_output = self.dropout(fused_output)  
        fused_output = fused_output.reshape(batch_size, -1)

        classification_output = self.output_layer(fused_output)

        return classification_output