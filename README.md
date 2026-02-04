# 音声言語モデルに関する講演資料

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org)

本リポジトリにて，講演スライド及びデモスクリプトを配布しています．

| 学会 | 日時 | スライド |
| --- | --- | --- |
| [音学シンポジウム2025](https://www.ipsj.or.jp/kenkyukai/event/mus143slp156.html) | 2025年6月13日 | [Link](slp2025-tutorial.pdf) |
| [6th joint ASA/ASJ meeting](https://acousticalsociety.org/honolulu-2025/) | 2025年12月3日 | [Link](asa2025-lecture.pdf) |
| [日本音響学会九州支部第3回オンラインセミナー](https://asj-kyushu.acoustics.jp) | 2026年2月27日 18:00-19:30 | 後日公開 |

## 質問

> demo2では、どのようにLlamaForSpeechLM-Instruct - Built with Llamaの事前学習モデルを行っていますか。詳細に教えていただきたいです。

ご質問ありがとうございます．[demo2.py](demo2.py)を用いて，下記の手順で事前学習を行っています．なお，学習にはNVIDIA RTX A6000 48GB VRAM GPUを1基用いました．
1. `sh scripts/download_clotho.sh`でClotho audio captioningデータセットをダウンロード
1. Whisper encoderとLlama 3.2 1Bを[2層MLPのadapter](demo2.py#L27)で接続．事前学習およびinstruction tuningを通して，WhisperおよびLlamaのパラメータを凍結し，adapterのみ更新
1. [train()](demo2.py#L421)を用いて，LibrispeechでのASRおよびClothoでのaudio captioningで事前学習
1. [generate_data()](demo2.py#L526)を用いて，[VITS](https://huggingface.co/kakao-enterprise/vits-vctk)でテキストベースのalpacaデータセットにおける入力テキストを音声合成し，音声入力の[alpacaデータセット](https://huggingface.co/datasets/ryota-komatsu/spoken-alpaca)を作成
1. [finetune()](demo2.py#L565)を用いて，作成したalpacaデータセットでcross-modal instruction tuning

## Setup

```shell
pip install -r requirements.txt
```

## Demo

### Phi-4-Multimodalで音声翻訳

[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ryota-komatsu/slp2025/blob/main/demo1.ipynb)

### Llama 3.2とWhisper encoderをadapterで接続してzero-shot instruction following

[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ryota-komatsu/slp2025/blob/main/demo2.ipynb)
[![model](https://img.shields.io/badge/%F0%9F%A4%97-Models-blue)](https://huggingface.co/ryota-komatsu/Llama-for-SpeechLM-Instruct)
[![dataset](https://img.shields.io/badge/%F0%9F%A4%97-Datasets-blue)](https://huggingface.co/datasets/ryota-komatsu/spoken-alpaca)

### Phonetic tokenとacoustic tokenとで再合成音声を比較

[![demo](https://img.shields.io/badge/Demo-blue)](https://ryota-komatsu.github.io/speech_resynth/)