# 音学シンポジウム 2025 チュートリアル 「マルチモーダル大規模言語モデル入門」

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org)

本リポジトリにて，[講演スライド](slp2025-tutorial.pdf)及びデモスクリプトを配布しています．
研究会詳細につきましては，下記Webページからご確認ください．

日時: 2025年6月13日 (金) 17:20-18:30 \
会場: 早稲田大学 西早稲田キャンパス \
詳細: [研究会Webページ](https://www.ipsj.or.jp/kenkyukai/event/mus143slp156.html)

## 質問
事前に[Google Forms](https://docs.google.com/forms/d/1pVKss5P4kh5-NQc-qp5K_dDA5_sxe0OUBAH18Nvzzew/edit?hl=ja)にいただいた質問には当日回答させていただきます．また，時間の許す限り，当日その場での質問も受け付けます．

## Demo

### Phi-4-Multimodalで音声翻訳

[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ryota-komatsu/slp2025/blob/main/demo1.ipynb)

### Llama 3.2とWhisper encoderをadapterで接続してzero-shot instruction following

[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ryota-komatsu/slp2025/blob/main/demo2.ipynb)
[![model](https://img.shields.io/badge/%F0%9F%A4%97-Models-blue)](https://huggingface.co/ryota-komatsu/Llama-for-SpeechLM-Instruct)
[![dataset](https://img.shields.io/badge/%F0%9F%A4%97-Datasets-blue)](https://huggingface.co/datasets/ryota-komatsu/spoken-alpaca)

### Phonetic tokenとacoustic tokenとで再合成音声を比較

[![demo](https://img.shields.io/badge/Demo-blue)](https://ryota-komatsu.github.io/speech_resynth/)