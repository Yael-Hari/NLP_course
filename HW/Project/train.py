import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import Dict
import time
import numpy as np
import random
import itertools
from model import Attention, Encoder, Decoder, Seq2Seq
from preprocessing import DeEnPairsData
from project_evaluate import compute_metrics


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True


def train_model_wrapper(
        train_loader: DataLoader,
        val_loader: DataLoader,
        input_dim: int,
        output_dim: int,
        enc_emb_dim: int,
        dec_emb_dim: int,
        enc_hid_dim: int,
        dec_hid_dim: int,
        enc_dropout: float,
        dec_dropout: float,
        n_epochs: int,
        clip: float,
        optimization,
) -> None:
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    print(f"Training on device: {device_str}")

    attention = Attention(enc_hid_dim, dec_hid_dim)
    encoder = Encoder(input_dim, enc_emb_dim, enc_hid_dim, dec_hid_dim, enc_dropout)
    decoder = Decoder(output_dim, dec_emb_dim, enc_hid_dim, dec_hid_dim, dec_dropout, attention)

    seq2seq_model = Seq2Seq(encoder, decoder, device).to(device)
    seq2seq_model.apply(init_weights)
    print(f'The model has {count_parameters(seq2seq_model):,} trainable parameters')
    criterion = nn.CrossEntropyLoss()  # ignore_index = TRG_PAD_IDX ??
    optimizer = optimization(seq2seq_model.parameters())

    best_valid_loss = float('inf')

    # ~~~~~~~~~~~~~~~~~~~~ TRAIN LOOP
    for epoch in range(n_epochs):

        start_time = time.time()

        train_loss, train_bleu = train(seq2seq_model, train_loader, optimizer, criterion, clip)
        val_loss, val_bleu = evaluate(seq2seq_model, val_loader, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            # torch.save(seq2seq_model.state_dict(), 'tut3-model.pt')  # TODO uncomment

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.4f} | Train BLEU: {train_bleu:7.3f}')
        print(f'\t\tVal. Loss: {val_loss:.4f}  |  Val. BLEU: {val_bleu:7.3f}')

    # ~~~~~~~~~~~~~~~~~~~~ PLOT
    # TODO


def train(model, loader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    epoch_bleu = 0

    for i, (src, trg) in enumerate(loader):

        optimizer.zero_grad()

        output = model(src, trg)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        bleu = compute_metrics(tagged_en=output, true_en=trg)
        epoch_bleu += bleu

    return epoch_loss / len(loader), epoch_bleu / len(loader)


def evaluate(model, loader, criterion):
    model.eval()

    epoch_loss = 0
    epoch_bleu = 0

    with torch.no_grad():
        for i, (src, trg) in enumerate(loader):
            output = model(src, trg, teacher_forcing_ratio=0)  # turn off teacher forcing

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            epoch_loss += loss.item()

            bleu = compute_metrics(tagged_en=output, true_en=trg)
            epoch_bleu += bleu

    return epoch_loss / len(loader), epoch_bleu / len(loader)


def init_weights(m):
    """ initialize all biases to zero and all weights from N~(0, 0.01) """
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    # constant params
    dataset = DeEnPairsData()
    train_loader, val_loader = dataset.get_tokens_labeled_data_loaders()
    input_dim = output_dim = len(train_loader)
    n_epochs = 10

    # changing params
    enc_emb_dim_l = [256]
    dec_emb_dim_l = [256]
    enc_hid_dim_l = [512]
    dec_hid_dim_l = [512]
    enc_dropout_l = [0.5]
    dec_dropout_l = [0.5]
    max_norm_clip_l = [1]
    optimization_l = [Adam]   # lr?

    cv_combinations = itertools.product(
        enc_emb_dim_l, dec_emb_dim_l, enc_hid_dim_l, dec_hid_dim_l,
        enc_dropout_l, dec_dropout_l, max_norm_clip_l, optimization_l
    )

    for args in cv_combinations:
        enc_emb_dim, dec_emb_dim, enc_hid_dim, dec_hid_dim, enc_dropout, dec_dropout, max_norm_clip, optimization = args
        print(f"=== curr model training: {enc_emb_dim=}, {dec_emb_dim=}, {enc_hid_dim=}, {dec_hid_dim=}, "
              f"{enc_dropout=}, {dec_dropout=}, {max_norm_clip=}, {optimization=} ===")

        train_model_wrapper(
            train_loader=train_loader,
            val_loader=val_loader,
            input_dim=input_dim,
            output_dim=output_dim,
            enc_emb_dim=enc_emb_dim,
            dec_emb_dim=dec_emb_dim,
            enc_hid_dim=enc_hid_dim,
            dec_hid_dim=dec_hid_dim,
            enc_dropout=enc_dropout,
            dec_dropout=dec_dropout,
            n_epochs=n_epochs,
            clip=max_norm_clip,
            optimization=optimization,
        )
