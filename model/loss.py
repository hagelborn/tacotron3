import torch.nn as nn

class Tacotron3Loss(nn.Module):
    def __init__(self):
        super(Tacotron3Loss, self).__init__()

    def forward(self,model_output,mel_target):
        mel_out, mel_out_postnet, _ = model_output
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
                   nn.MSELoss()(mel_out_postnet, mel_target)
        return mel_loss