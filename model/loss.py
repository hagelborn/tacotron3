import torch.nn as nn

class Tacotron3Loss(nn.Module):
    def __init__(self):
        super(Tacotron3Loss, self).__init__()

    def forward(self,model_output,target):
        mel_out, mel_out_postnet, _, label = model_output
        mel_target, label_target = target
        label_target.unsqueeze_(1)

        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
                   nn.MSELoss()(mel_out_postnet, mel_target)
        label_loss = nn.functional.binary_cross_entropy(label,label_target)
        return mel_loss + label_loss