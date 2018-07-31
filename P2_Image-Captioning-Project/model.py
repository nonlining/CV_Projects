import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.batchNorm = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.batchNorm(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers = 2, dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,  dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, features, captions):
        captions = captions[:, :-1]
        captions_embedding = self.embed(captions)
        #print(captions_embedding.shape)
        #print("image features ",features.unsqueeze(1).shape)
        feature_embedding_inputs = torch.cat((features.unsqueeze(1), captions_embedding), 1)
        #print(feature_embedding_inputs.shape)

        lstm_output, _ = self.lstm(feature_embedding_inputs, None)
        #print(lstm_output.shape)
        outputs = self.linear(lstm_output)
        #print(outputs.shape)
        return outputs

    def sample(self, inputs, states=None, max_length = 20, stop_idx=1):

        sentence = []
        state = None
        for _ in range(max_length):
            lstm_out, state = self.lstm(inputs, state)
            output = self.linear(lstm_out)
            predict = torch.argmax(output, dim=2)
            predict_index = predict.item()
            sentence.append(predict_index)

            if predict_index == stop_idx:
                break

            inputs = self.embed(predict)

        return sentence