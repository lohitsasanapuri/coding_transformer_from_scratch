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

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encode(source.long(), source_mask)
    decoder_input = torch.full((1, 1), sos_idx, dtype=torch.long, device=device)

    while decoder_input.size(1) < max_len:
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        # order: (tgt, encoder_output, src_mask, tgt_mask)
        decoder_output = model.decode(decoder_input, encoder_output, source_mask, decoder_mask)
        logits_last = model.projection(decoder_output[:, -1])
        _, next_word = torch.max(logits_last, dim=1)

        next_token = next_word.item()
        decoder_input = torch.cat(
            [decoder_input, torch.tensor([[next_token]], dtype=torch.long, device=device)],
            dim=1
        )
        if next_token == eos_idx:
            break

    return decoder_input.squeeze(0)



def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, device,
                   print_msg, globel_state, writer, num_examples=2):
    model.eval()
    count = 0
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device).long()
            encoder_mask  = batch['encoder_mask'].to(device)

            # decode prediction
            model_output_ids = greedy_decode(
                model, encoder_input, encoder_mask,
                tokenizer_src, tokenizer_tgt, validation_ds.dataset.seq_len, device
            )

            # ✅ use keys that your Dataset actually returns
            source_text = batch['src_text'][0]  # NOT 'lang_src'
            target_text = batch['tgt_text'][0]  # NOT 'lang_tgt'

            # pretty print
            model_out_text = tokenizer_tgt.decode(model_output_ids.detach().cpu().tolist())

            print_msg('-' * console_width)
            print_msg(f"Source:    {source_text}")
            print_msg(f"Target:    {target_text}")
            print_msg(f"Predicted: {model_out_text}")

            if count >= num_examples:
                break




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
    
    # for quick testing use a smaller dataset
    ds_raw = ds_raw.select(range(160))

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
        #model.train()
        batch_iterator = tqdm(train_dataloader, desc=f' Processing epoch {epoch+1}/{config["num_epochs"]}' )
        for batch in batch_iterator:
            model.train()
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
        
        run_validation(
                model, val_dataloader, tokenizer_src, tokenizer_tgt, device,
                lambda msg: batch_iterator.write(msg),   # print_msg
                device,                                  # globel_state (if you need it)
                writer,
                num_examples=2)
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