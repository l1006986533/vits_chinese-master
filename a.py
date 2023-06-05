import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io import wavfile
from scipy.io.wavfile import write

from flask import Flask, request, send_from_directory
app = Flask(__name__)

# device = torch.device("cpu")
# model.to(device)
def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


hps = utils.get_hparams_from_file("configs/config.json")

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model)
_ = net_g.eval()

# _ = utils.load_checkpoint("logs/woman_csmsc/G_100000.pth", net_g, None) 
_ = utils.load_checkpoint("logs/G_94000.pth", net_g, None) 

@app.route('/')
def hello_world():
    return '<head><meta charset="utf-8"></head><form action="http://127.0.0.1:5000/text" style="zoom:200%">输入一段文字：(仅限中文)<br><input name="text" style="width:300px" value="廖一强音色语音合成测试"><br><input type="submit"></form>'

@app.route('/text')
def webtext():
    text = request.args.get("text")
    stn_tst = get_text(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])

        # x_tst = stn_tst.cpu().unsqueeze(0)
        # x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cpu()

        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()

    sampling_rate = 44100
    wavfile.write('temp/abc1.wav', sampling_rate, audio)
    return send_from_directory("temp",'abc1.wav')
# ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))

if __name__ == '__main__':
   app.run()