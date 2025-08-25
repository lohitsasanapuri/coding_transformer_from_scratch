import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token =  torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token =  torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token =  torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_inputs_token = self.tokenizer_src.encode(src_text).ids
        dec_inputs_token = self.tokenizer_tgt.encode(tgt_text).ids

        # padding token counts
        enc_num_pad_tokens = self.seq_len - len(enc_inputs_token) - 2
        dec_num_pad_tokens = self.seq_len - len(dec_inputs_token) - 1

        if enc_num_pad_tokens < 0 or dec_num_pad_tokens < 0:
            raise ValueError(f" sentence too long ")
        
        # Add SOS and EOS tokens and padding 
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_inputs_token, dtype=torch.int64),
                self.eos_token,
                self.pad_token.repeat(enc_num_pad_tokens)
            ]
        )

        # Add SOS token and padding
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_inputs_token, dtype=torch.int64),
                self.pad_token.repeat(dec_num_pad_tokens)
            ]
        )
        # Add EOS token and padding
        label = torch.cat(
            [
                torch.tensor(dec_inputs_token, dtype=torch.int64),
                self.eos_token,
                self.pad_token.repeat(dec_num_pad_tokens)
            ]

        )

        assert len(encoder_input) == self.seq_len
        assert len(decoder_input) == self.seq_len
        assert len(label) == self.seq_len

        return {
            'encoder_input': encoder_input, # (seq len)
            'decoder_input': decoder_input,# (seq len)
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, seq len)
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),#(1, seq_len) & (1, seq_len, seq len)
            'label': label,
            'src_text': src_text,
            'tgt_text': tgt_text
        }
def causal_mask(size):
    mask = torch.tril(torch.ones((1, size, size), dtype=torch.int64))
    return mask


