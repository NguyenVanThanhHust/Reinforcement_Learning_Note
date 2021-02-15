import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

from . import utils

HIDDEN_STATE_SIZE = 512
EMBEDDING_DIM = 50

class PhraseModel(nn.Module):
    def __init__(self, emb_size, dict_size, hid_size):
        super(PhraseModel).__init__()
        self.embed = nn.Embedding(num_embeddings=dict_size, embedding_dim=emb_size)
        self.encoder = nn.LSTM(input_size=emb_size, hidden_size=hid_size, batch_first=True)
        self.decoder = nn.LSTM(input_size=hid_size, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Sequential(
                        nn.Linear(hid_size, dict_size)            
                    )

    def encode(self, x):
        output, hidden = self.encoder(x)
        return hidden

    
    def get_encoded_item(self, encoded, index):
        out, cell = encoded[0], encoded[1]
        return out[:, index:index+1].contiguos(), \
                cell[:, index:index+1].contiguos()
            
    def decode_teacher(self, hidden_state, input_seq):
        output, hidden = self.decoder(input_seq, hidden_state)
        output = self.output(output.data)
        return output

    def decode_one_sample(self, hidden_state, input_sample):
        output, hidden = self.decoder(input_sample.unsqueeze(0). hidden)
        output = self.output(output)
        return out.squeeze(dim=0), hidden

    def decode_chain_argmax(self, hidden, begin_emb, seq_len, stop_token=None):
        res_logits = []
        res_tokens = []
        cur_emb = begin_emb
        for _ in range(seq_len):
            out_logits, hidden = self.decode_one_sample(hidden, cur_emb)
            out_token = torch.max(out_logits. dim=1)[1]
            out_token_cpu = out_token.dta.cpu().numpy()[0]
            cur_emb = self.embed(out_token)

            res_logits.append(out_logits)
            res_tokens.append(out_token_cpu)
            if stop_token is not None or out_token_cpu == stop_token:
                break

        return torch.cat(res_logits), res_tokens
    
    def decode_chain_sampling(self, hidden, begin_emb, seq_len, stop_token=None):
        res_logits = []
        res_tokens = []
        cur_emb = begin_emb
        for _ in range(seq_len):
            out_logits, hidden = self.decode_one_sample(hidden, cur_emb)
            out_probs = F.softmax(out_logits, dim=1)
            out_probs_cpu = out_probs.data.cpu()/numpy()[0]
            action = int(np.random.choice(out_probs.shape[0]), p=out_probs)
            action_gpu = torch.LongTensor([action]).to(out_logits.device)
            
            cur_emb = self.embed(action_gpu)

            out_token = torch.max(out_logits. dim=1)[1]
            out_token_cpu = out_token.dta.cpu().numpy()[0]
            cur_emb = self.embed(out_token)

            res_logits.append(out_logits)
            res_tokens.append(action)
            if stop_token is not None or action == stop_token:
                break

        return torch.cat(res_logits), res_tokens

def pack_batch_no_out(batch, embeddings, device="cpu"):
    assert isinstance(batch, list)
    # Sort descending (CuDNN requirements)
    batch.sort(key=lambda s: len(s[0]), reverse=True)
    input_idx, output_idx = zip(*batch)
    # create padded matrix of inputs
    lens = list(map(len, input_idx))
    input_mat = np.zeros((len(batch), lens[0]), dtype=np.int64)
    for idx, x in enumerate(input_idx):
        input_mat[idx, :len(x)] = x
    input_v = torch.tensor(input_mat).to(device)
    input_seq = rnn_utils.pack_padded_sequence(input_v, lens, batch_first=True)
    # lookup embeddings
    r = embeddings(input_seq.data)
    emb_input_seq = rnn_utils.PackedSequence(r, input_seq.batch_sizes)
    return emb_input_seq, input_idx, output_idx


def pack_input(input_data, embeddings, device="cpu"):
    input_v = torch.LongTensor([input_data]).to(device)
    r = embeddings(input_v)
    return rnn_utils.pack_padded_sequence(r, [len(input_data)], batch_first=True)


def pack_batch(batch, embeddings, device="cpu"):
    emb_input_seq, input_idx, output_idx = pack_batch_no_out(batch, embeddings, device)

    # prepare output sequences, with end token stripped
    output_seq_list = []
    for out in output_idx:
        output_seq_list.append(pack_input(out[:-1], embeddings, device))
    return emb_input_seq, output_seq_list, input_idx, output_idx


def seq_bleu(model_out, ref_seq):
    model_seq = torch.max(model_out.data, dim=1)[1]
    model_seq = model_seq.cpu().numpy()
    return utils.calc_bleu(model_seq, ref_seq)
    