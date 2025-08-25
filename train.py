import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from dataset import BilingualDataset, causal_mask
from model  import build_transformer
from config import get_config, get_weights_file_path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
# reltive paths to absolute paths
from pathlib import Path

import warnings

from torch.utils.tensorboard import SummaryWriter 

from tqdm import tqdm

def get_all_sentences(ds, lang):
    for item in ds :
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    # config['tokenizer_file] = '../tokenizers/tokenizer_{0}.json'
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token= "[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]","[SOS]","[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

    # build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # keep 90% for training and 10% for validation
    train_ds_size  = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    ds_train, ds_val = torch.utils.data.random_split(ds_raw, [train_ds_size, val_ds_size]) 

    train_ds = BilingualDataset(ds_train, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(ds_val, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])  

    max_src_len = 0
    max_tgt_len = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_src_len, len(src_ids))
        max_len_tgt = max(max_tgt_len, len(tgt_ids))
    
    print(f"Max source length: {max_len_src}")
    print(f"Max target length: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['d_model'])
    return model



def train_model(config):

    # define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # make sure weights folder exists
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config) 

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    # optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], eps=1e-9)

    # resume training
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_path = get_weights_file_path(config, config['preload'])
        print(f"Loading model weights from {model_path}")
        state = torch.load(model_path)
        initial_epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f' Processing epoch {epoch+1}/{config["num_epochs"]}' )
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (batch_size, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (batch_size, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch_size, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (batch_size, 1, seq_len, seq_len)
            
            # run the tensoer through the model

            encoder_output  = model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
            decoder_output  = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask) # (batch_size, seq_len, d_model)
            projected_output = model.projection(decoder_output) # (batch_size, seq_len, tgt_vocab_size)
            labels = batch['label'].to(device) # (batch_size, seq_len)

            # compute loss
            # ( batch_size, seq_len, tgt_vocab_size) --> (batch_size*seq_len, tgt_vocab_size)
            loss = loss_fn(projected_output.view(-1, tokenizer_tgt.get_vocab_size()), labels.view(-1)).to(device)

            batch_iterator.set_postfix_str(f"loss: {loss.item():.4f}")

            # log the loss
            writer.add_scalar('Training loss', loss.item(), global_step)
            writer.flush()

            # backprop and update the weights
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()   
            global_step += 1
        # save the model weights
        model_filename = get_weights_file_path(config, f'{epoch+1:03d}')
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)