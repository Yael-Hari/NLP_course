import numpy as np
import torch


def predict_LSTM(
    LSTM_model,
    num_epochs: int,
    test_loader=None,
):

    # -------
    # GPU
    # -------
    # First checking if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Training on GPU.")
    else:
        print("No GPU available, training on CPU.")
    LSTM_model.to(device)

    # ----------------------------------
    # Epoch Loop
    # ----------------------------------
    for epoch_num in range(num_epochs):
        for batch_num, (sentences, sen_lengths) in enumerate(test_loader):
            # if training on gpu
            sentences, sen_lengths = (
                sentences.to(device),
                sen_lengths.to(device),
            )

            # forward
            outputs = LSTM_model(sentences, sen_lengths)

            # predictions
            preds = outputs.argmax(dim=1).clone().detach().cpu()
            y_pred = np.array(preds.view(-1))

    return y_pred
