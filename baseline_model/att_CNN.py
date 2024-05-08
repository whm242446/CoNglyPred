import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y
    

# multihead-attention
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

        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size) 
        x = x.view(*new_x_shape)   
        return x.permute(0, 2, 1, 3) 


    def forward(self, output):
        mixed_query_layer = self.query(output) 
        mixed_key_layer = self.key(output)
        mixed_value_layer = self.value(output)

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
    



# class CrossAttention(nn.Module):
    
#     def __init__(self, hidden_size, num_heads, dropout_prob):   
#         super(CrossAttention, self).__init__()
#         if hidden_size % num_heads != 0: 
#             raise ValueError(
#                 "The hidden size (%d) is not a multiple of the number of attention "
#                 "heads (%d)" % (hidden_size, num_heads))
            
#         self.num_heads = num_heads
#         self.attention_head_size = int(hidden_size / num_heads)
#         self.all_head_size = int(self.num_heads * self.attention_head_size)   

#         self.query = nn.Linear(hidden_size, self.all_head_size) # 128, 128
#         self.key = nn.Linear(hidden_size, self.all_head_size)
#         self.value = nn.Linear(hidden_size, self.all_head_size)
        
#         # dropout
#         self.dropout = nn.Dropout(dropout_prob)

#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size) # [bs, seqlen, 8, 16]
#         x = x.view(*new_x_shape)   
#         return x.permute(0, 2, 1, 3)   # [bs, 8, seqlen, 16]


#     def forward(self, Q_ouput, K_output):
#         mixed_query_layer = self.query(Q_ouput)
#         mixed_key_layer = self.key(K_output) 
#         mixed_value_layer = self.value(K_output)

#         query_layer = self.transpose_for_scores(mixed_query_layer) 
#         key_layer = self.transpose_for_scores(mixed_key_layer) 
#         value_layer = self.transpose_for_scores(mixed_value_layer)

#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
#         # [bs, 8, seqlen, 16]*[bs, 8, 16, seqlen]  ==> [bs, 8, seqlen, seqlen]
#         attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#         print(attention_scores)
#         print(attention_scores.shape)
#         attention_probs = nn.Softmax(dim=-1)(attention_scores)
#         #print(attention_probs)

#         attention_probs = self.dropout(attention_probs)
        
#         context_layer = torch.matmul(attention_probs, value_layer) 
#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#         attention_layer = context_layer.view(*new_context_layer_shape)
        
#         return attention_layer 

class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_prob):
        super(CrossAttention, self).__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention heads (%d)" % (hidden_size, num_heads))
        
        self.num_heads = num_heads
        self.attention_head_size = int(hidden_size / num_heads)
        self.all_head_size = self.num_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(dropout_prob)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, Q_output, K_output):
        mixed_query_layer = self.query(Q_output)
        mixed_key_layer = self.key(K_output)
        mixed_value_layer = self.value(K_output)
        
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


class AttentionClassifier(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_prob, num_layers):
        super(AttentionClassifier, self).__init__()
        self.layers = nn.ModuleList([
            SelfAttention(hidden_size, num_heads, dropout_prob)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    
class CNNAttentionLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, hidden_size, num_heads, dropout_prob):
        super(CNNAttentionLayer, self).__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size, padding=kernel_size // 2)
        self.linear = nn.Linear(output_channels, hidden_size)
        self.self_attention = SelfAttention(hidden_size, num_heads, dropout_prob)

    def forward(self, x):
        x = x.transpose(1, 2) 
        conv_output = self.conv(x) 
        conv_output = conv_output.transpose(1, 2) 
        attention_input = self.linear(conv_output) 

        attention_output = self.self_attention(attention_input) 

        return attention_output



class AttentionCNNLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_prob, kernel_size, output_channels):
        super(AttentionCNNLayer, self).__init__()
        self.self_attention = SelfAttention(hidden_size, num_heads, dropout_prob)
        
        self.conv = nn.Conv1d(hidden_size, output_channels, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        attention_output = self.self_attention(x)
        conv_input = attention_output.transpose(1, 2)  
        conv_output = self.conv(conv_input)

        return conv_output


"""
    MLP Layer used after graph vector representation
"""

class MLPLayer(nn.Module):

    def __init__(self, input_dim, output_dim, L=2): 
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y

