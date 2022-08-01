import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import pad_sequence
<<<<<<< HEAD

__all__ = ["DSN"]


class PositionalEncoding(nn.Module):
=======
import pdb

__all__ = ['DSN']


class PositionalEncoding(nn.Module):

>>>>>>> d1a96e10480e3d10294c2ef1b61a8f5361e362ad
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
<<<<<<< HEAD
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self, out_dim=512, in_dim=1024, nhead=4, nhid=2048, nlayers=3, dropout=0.1
    ):
=======
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self, out_dim=512, in_dim=1024, nhead=4, nhid=2048, nlayers=3, dropout=0.1):
>>>>>>> d1a96e10480e3d10294c2ef1b61a8f5361e362ad
        super(TransformerModel, self).__init__()

        self.pos_encoder = PositionalEncoding(in_dim, dropout)
        encoder_layers = TransformerEncoderLayer(in_dim, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(in_dim, out_dim)
<<<<<<< HEAD

=======
>>>>>>> d1a96e10480e3d10294c2ef1b61a8f5361e362ad
    #     self.init_weights()
    #
    #
    # def init_weights(self):
    #     initrange = 0.1
    #     self.decoder.bias.data.zero_()
    #     self.decoder.weight.data.uniform_(-initrange, initrange)

<<<<<<< HEAD
    def forward(self, src, src_mask=None):
        src = self.pos_encoder(src)
        feature = self.transformer_encoder(src, mask=src_mask)
=======
    def forward(self, src,src_mask=None):
        src = self.pos_encoder(src)
        feature = self.transformer_encoder(src,mask=src_mask)
>>>>>>> d1a96e10480e3d10294c2ef1b61a8f5361e362ad
        output = self.decoder(feature)
        return output, feature


class DSN(nn.Module):
    """Deep Summarization Network"""
<<<<<<< HEAD

    def __init__(self, hps):
        super(DSN, self).__init__()
        self.trans1 = TransformerModel(
            out_dim=hps.shot_dim,
            in_dim=1024,
            nhead=hps.nhead1,
            nhid=hps.nhid1,
            nlayers=hps.nlayers1,
            dropout=hps.dropout1,
        )
        self.trans2 = TransformerModel(
            out_dim=hps.shot_dim,
            in_dim=hps.shot_dim,
            nhead=hps.nhead2,
            nhid=hps.nhid2,
            nlayers=hps.nlayers2,
            dropout=hps.dropout2,
        )

        self.fused = nn.Linear(hps.shot_dim * 2, 1)
        self.extra = nn.Embedding(1, 1024)
        self.extra_shot = nn.Embedding(1, hps.shot_dim)

    def postprocess(self, p, boundary):
        if len(p) == 0:
            return
        temp = [
            p[i : i + 1].expand(boundary[i, 1] - boundary[i, 0] + 1)
            for i in range(p.shape[0])
        ]
        probs = torch.cat(temp, 0)
        return probs[::15]

    def forward(self, inputs1, inputs2=None):
        if self.training:
            shots1, boundary1 = inputs1["shots"], inputs1["boundary"]
            shots2, boundary2 = inputs2["shots"], inputs2["boundary"]
=======
    def __init__(self, hps):
        super(DSN, self).__init__()
        self.trans1 = TransformerModel(out_dim=hps.shot_dim, in_dim=1024, nhead=hps.nhead1, nhid=hps.nhid1,
                                       nlayers=hps.nlayers1, dropout=hps.dropout1)
        self.trans2 = TransformerModel(out_dim=hps.shot_dim, in_dim=hps.shot_dim, nhead=hps.nhead2, nhid=hps.nhid2,
                                       nlayers=hps.nlayers2, dropout=hps.dropout2)

        self.fused = nn.Linear(hps.shot_dim*2, 1)
        self.extra = nn.Embedding(1, 1024)
        self.extra_shot = nn.Embedding(1, hps.shot_dim)


    def postprocess(self,p,boundary):
        if len(p)==0: return
        temp = [p[i:i+1].expand(boundary[i,1]-boundary[i,0]+1) for i in range(p.shape[0])]
        probs = torch.cat(temp, 0)
        return probs[::15]

    def forward(self, inputs1,inputs2=None):
        if self.training:
            shots1, boundary1 = inputs1['shots'], inputs1['boundary']
            shots2, boundary2 = inputs2['shots'], inputs2['boundary']
>>>>>>> d1a96e10480e3d10294c2ef1b61a8f5361e362ad
            n_shots1 = len(shots1)
            shots = shots1 + shots2
            shots = [torch.cat([self.extra.weight, shot], 0) for shot in shots]
            shots = pad_sequence(shots)

            h, _ = self.trans1(shots)
<<<<<<< HEAD
            h = h[0, :, :].unsqueeze(1)
            seq1, seq2 = h[:n_shots1], h[n_shots1:]
            token1 = token2 = self.extra_shot.weight.unsqueeze(0)
            seq = torch.cat([token1, seq1, seq2, token2], 0)
            src_mask = torch.zeros((len(seq), len(seq)))
            src_mask[0, n_shots1 + 1 :] = 1
            src_mask[-1, 0 : n_shots1 + 1] = 1
            src_mask = src_mask.bool().to(seq.device)
            decoded, feature = self.trans2(seq, src_mask=src_mask)
            decoded_seq1, decoded_seq2 = (
                decoded[1 : n_shots1 + 1],
                decoded[n_shots1 + 1 : -1],
            )
            decoded_token1, decoded_token2 = decoded[0:1].expand_as(
                decoded_seq1
            ), decoded[-1:].expand_as(decoded_seq2)
            decoded_seq1, decoded_seq2 = torch.cat(
                (decoded_seq1, decoded_token1), 2
            ), torch.cat((decoded_seq2, decoded_token2), 2)
            decoded_seq = torch.cat((decoded_seq1, decoded_seq2), 0)
=======
            h = h[0,:,:].unsqueeze(1)
            seq1, seq2 = h[:n_shots1], h[n_shots1:]
            token1 = token2 = self.extra_shot.weight.unsqueeze(0)
            seq = torch.cat([token1,seq1,seq2,token2],0)
            src_mask = torch.zeros((len(seq),len(seq)))
            src_mask[0,n_shots1 + 1:] = 1
            src_mask[-1,0:n_shots1+1] = 1
            src_mask = src_mask.bool().to(seq.device)
            decoded, feature = self.trans2(seq,src_mask=src_mask)
            decoded_seq1, decoded_seq2 = decoded[1:n_shots1 + 1], decoded[n_shots1 + 1:-1]
            decoded_token1,decoded_token2 = decoded[0:1].expand_as(decoded_seq1),decoded[-1:].expand_as(decoded_seq2)
            decoded_seq1, decoded_seq2 = torch.cat((decoded_seq1, decoded_token1), 2), torch.cat(
                (decoded_seq2, decoded_token2), 2)
            decoded_seq = torch.cat((decoded_seq1, decoded_seq2),0)
>>>>>>> d1a96e10480e3d10294c2ef1b61a8f5361e362ad
            probs = self.fused(decoded_seq).squeeze()
            p1, p2 = probs[:n_shots1], probs[n_shots1:]
            feature_video1 = feature[0]
            weighted_shots1 = seq1 * p1.unsqueeze(1).unsqueeze(1)
<<<<<<< HEAD
            weighted_shots = torch.cat(
                [self.extra_shot.weight.unsqueeze(0), weighted_shots1], 0
            )
            _, feature_key = self.trans2(weighted_shots)
            feature_summary = feature_key[0]
            rec_loss = (feature_video1 - feature_summary).pow(2).mean()
            p1, p2 = self.postprocess(p1, boundary1), self.postprocess(p2, boundary2)
            return p1, p2, rec_loss
        else:
            shots, boundary1 = inputs1["shots"], inputs1["boundary"]
            n_shots1 = len(shots)
            shots = [torch.cat([self.extra.weight, shot], 0) for shot in shots]
            shots = pad_sequence(shots)
            h, _ = self.trans1(shots)
            h = h[0, :, :].unsqueeze(1)
            seq1 = h[:n_shots1]
            seq = torch.cat([self.extra_shot.weight.unsqueeze(0), seq1], 0)
            decoded, _ = self.trans2(seq)
            decoded_seq1 = decoded[1 : n_shots1 + 1]
            decoded_token1 = decoded[0:1].expand_as(decoded_seq1)
            decoded_seq1 = torch.cat((decoded_seq1, decoded_token1), 2)
            probs = self.fused(decoded_seq1).squeeze()
            p1 = self.postprocess(probs, boundary1)
=======
            weighted_shots = torch.cat([self.extra_shot.weight.unsqueeze(0), weighted_shots1], 0)
            _, feature_key = self.trans2(weighted_shots)
            feature_summary = feature_key[0]
            rec_loss = (feature_video1 - feature_summary).pow(2).mean()
            p1, p2 = self.postprocess(p1,boundary1), self.postprocess(p2, boundary2)
            return p1, p2, rec_loss
        else:
            shots, boundary1 = inputs1['shots'], inputs1['boundary']
            n_shots1 = len(shots)
            shots = [torch.cat([self.extra.weight, shot], 0) for shot in shots]
            shots = pad_sequence(shots)
            h,_ = self.trans1(shots)
            h = h[0,:,:].unsqueeze(1)
            seq1 = h[:n_shots1]
            seq = torch.cat([self.extra_shot.weight.unsqueeze(0),seq1],0)
            decoded,_ = self.trans2(seq)
            decoded_seq1 = decoded[1:n_shots1 + 1]
            decoded_token1 = decoded[0:1].expand_as(decoded_seq1)
            decoded_seq1 = torch.cat((decoded_seq1, decoded_token1), 2)
            probs = self.fused(decoded_seq1).squeeze()
            p1 = self.postprocess(probs,boundary1)
>>>>>>> d1a96e10480e3d10294c2ef1b61a8f5361e362ad
            return p1
