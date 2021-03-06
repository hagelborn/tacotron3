




################################
# Experiment Parameters        #
################################
epochs=1000
iters_per_checkpoint=200
seed=1234
dynamic_loss_scaling=True
fp16_run=False
cudnn_enabled=False
cudnn_benchmark=False
ignore_layers=['embedding.weight']

################################
# Data Parameters             #
################################

seq_len = 160
max_len = 850 #

################################
# Audio Parameters             #
################################
max_wav_value=32768.0
sampling_rate=16000
filter_length=1024
hop_length=256
win_length=1024
n_mel_channels=256
mel_fmin=0.0
mel_fmax=8000.0

################################
# Model Parameters             #
################################

# Encoder parameters
encoder_num_layers = 1
encoder_hidden_dim = 5
embedding_dim = 64
bidirect = False
encoder_embedding_dim = embedding_dim + encoder_hidden_dim*(1+bidirect)


# Decoder parameters
n_frames_per_step=1  # currently only 1 is supported
decoder_rnn_dim=1024
prenet_dim=256
max_decoder_steps=1000
gate_threshold=0.5
p_attention_dropout=0.1
p_decoder_dropout=0.1

# Attention parameters
attention_rnn_dim=1024
attention_dim=128

# Location Layer parameters
attention_location_n_filters=32
attention_location_kernel_size=31

# Mel-post processing network parameters
postnet_embedding_dim=512
postnet_kernel_size=5
postnet_n_convolutions=5

################################
# Optimization Hyperparameters #
################################
use_saved_learning_rate=False,
learning_rate=5e-4
weight_decay=1e-6
grad_clip_thresh=1.0
batch_size=8
mask_padding=True  # set model's padded outputs to padded values


################################
# Validation parameters        #
################################
val_batch_size = 6

