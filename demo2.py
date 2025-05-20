import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from datasets import Audio, load_dataset
from torch import nn
from torch.utils.data import ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    WhisperForConditionalGeneration,
)


class Adapter(nn.Module):
    def __init__(
        self,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        kernel_size: int,
        bias: bool,
    ):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size)
        self.linear1 = nn.Linear(encoder_hidden_size, 2 * decoder_hidden_size, bias=bias)
        self.linear2 = nn.Linear(2 * decoder_hidden_size, decoder_hidden_size, bias=bias)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.pool(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.linear1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class LlamaForSpeechLMConfig(PretrainedConfig):
    model_type = "llama_for_speech_lm"

    def __init__(
        self,
        encoder_id: str = "openai/whisper-small.en",
        decoder_id: str = "meta-llama/Llama-3.2-1B-Instruct",
        adapter_kernel_size: int = 4,
        adapter_linear_bias: bool = False,
        **kwargs,
    ):
        self.encoder_id = encoder_id
        self.decoder_id = decoder_id
        self.adapter_kernel_size = adapter_kernel_size
        self.adapter_linear_bias = adapter_linear_bias
        super().__init__(**kwargs)


class LlamaForSpeechLM(PreTrainedModel):
    config_class = LlamaForSpeechLMConfig
    _tied_weights_keys = ["decoder.lm_head.weight"]

    def __init__(self, config: LlamaForSpeechLMConfig):
        super().__init__(config)
        self.encoder = WhisperForConditionalGeneration.from_pretrained(config.encoder_id).model.encoder
        self.decoder = AutoModelForCausalLM.from_pretrained(config.decoder_id, torch_dtype=torch.bfloat16)
        self.adapter = Adapter(
            self.encoder.config.d_model,
            self.decoder.config.hidden_size,
            config.adapter_kernel_size,
            config.adapter_linear_bias,
        )

        self.encoder.requires_grad_(False)
        self.decoder.requires_grad_(False)

    def get_input_embeddings(self):
        return self.decoder.model.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.decoder.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.decoder.lm_head = new_embeddings

    def embed(
        self,
        input_features: torch.FloatTensor,
        input_ids: torch.LongTensor,
        encoder_attention_mask: torch.LongTensor,
        decoder_attention_mask: torch.LongTensor,
    ):
        encoder_outputs = self.encoder(input_features)
        encoder_hidden_states = encoder_outputs[0]

        lengths = self.encoder._get_feat_extract_output_lengths(encoder_attention_mask.sum(dim=1, keepdim=True))
        lengths = lengths // self.config.adapter_kernel_size
        max_len = lengths.max()

        encoder_hidden_states = self.adapter(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states[:, :max_len]

        inputs_embeds = self.decoder.model.embed_tokens(input_ids)
        inputs_embeds = torch.cat((encoder_hidden_states, inputs_embeds), dim=1)

        attention_mask = torch.cat(
            (
                (
                    torch.arange(encoder_hidden_states.shape[1], device=decoder_attention_mask.device).unsqueeze(0)
                    < lengths
                ).long(),
                decoder_attention_mask,
            ),
            dim=1,
        )
        return inputs_embeds, attention_mask

    def forward(
        self,
        input_features: torch.FloatTensor,
        input_ids: torch.LongTensor,
        encoder_attention_mask: torch.LongTensor,
        decoder_attention_mask: torch.LongTensor,
    ):
        """
        Args:
            input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, feature_length)`):
                Log mel spectrogram.
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Token ids.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, feature_length)`):
                1: non-mask
                0: mask
            decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                1: non-mask
                0: mask
        """
        inputs_embeds, attention_mask = self.embed(
            input_features, input_ids, encoder_attention_mask, decoder_attention_mask
        )

        labels = F.pad(input_ids, (inputs_embeds.shape[1] - input_ids.shape[1], 0), value=-100)

        decoder_outputs = self.decoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return decoder_outputs.loss

    @torch.amp.autocast("cuda", dtype=torch.bfloat16)
    @torch.no_grad()
    def generate(
        self,
        input_features: torch.FloatTensor,
        input_ids: torch.LongTensor,
        encoder_attention_mask: torch.LongTensor,
        decoder_attention_mask: torch.LongTensor,
        **kwargs,
    ):
        inputs_embeds, attention_mask = self.embed(
            input_features, input_ids, encoder_attention_mask, decoder_attention_mask
        )

        generated_ids = self.decoder.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)
        return generated_ids


class Clotho(torch.utils.data.Dataset):
    def __init__(self, root="data", split: str = "development", caption_idx: int = 1):
        """
        Args:
            split: development | validation | evaluation
        """
        self.audio_dir = os.path.join(root, "clotho", split)
        caption_path = os.path.join(root, f"clotho/clotho_captions_{split}.csv")

        self.captions = pd.read_csv(caption_path, encoding="ISO-8859-1")
        self.caption_idx = caption_idx

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, n: int) -> Tuple[torch.FloatTensor, int, str]:
        """
        Returns:
            audio: 15 to 30 seconds duration
            caption: 8 to 20 words length
        """
        item = self.captions.iloc[n]  # file_name,caption_1,caption_2,caption_3,caption_4,caption_5

        audio_path = os.path.join(self.audio_dir, item["file_name"])
        audio, sr = torchaudio.load(audio_path)
        audio = torchaudio.functional.resample(audio, sr, 16000)

        caption = item[f"caption_{self.caption_idx}"]

        return audio, 16000, caption


def get_lr_schedule(
    optimizer,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_schedule(current_step: int) -> float:
        if current_step < warmup_steps:
            return (min_lr + (base_lr - min_lr) * current_step / warmup_steps) / base_lr
        else:
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            return (min_lr + (base_lr - min_lr) * (1 - progress)) / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)


def _train(
    model: LlamaForSpeechLM,
    loader: torch.utils.data.DataLoader,
    lr: float = 1e-3,
    epoch: int = 1,
    warmup_steps: int = 10,
    init_grad_scale: float = 1e32,
    clip_grad_norm: float = 1.0,
    grad_accumulation: int = 128,
    model_dir="models/Llama-for-SpeechLM",
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # learning rate scheduler
    lr_scheduler = get_lr_schedule(
        optimizer,
        len(loader) // grad_accumulation * epoch,
        warmup_steps,
        lr,
        lr * 0.1,
    )

    scaler = torch.amp.GradScaler("cuda", init_scale=init_grad_scale)
    writer = SummaryWriter()

    step = 0

    for epoch in range(1, epoch + 1):
        model.train()

        for batch_idx, batch in enumerate(tqdm(loader, desc=f"epoch {epoch}")):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch)
                loss = loss / grad_accumulation
            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accumulation == 0:
                # gradient clipping
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

                # update
                scaler.step(optimizer)
                scale = scaler.get_scale()
                scaler.update()
                optimizer.zero_grad()

                # update learning rate
                lr = lr_scheduler.get_last_lr()[0]
                lr_scheduler.step()

                step += 1

                # tensorboard log
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/lr", lr, step)
                writer.add_scalar("train/scale", scale, step)
                writer.add_scalar("train/grad_norm", grad_norm.item(), step)

        Path(model_dir).parent.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_dir)


def train(
    encoder_id="openai/whisper-small.en",
    decoder_id="meta-llama/Llama-3.2-1B-Instruct",
    batch_size: int = 4,
    lr: float = 1e-3,
    epoch: int = 1,
    warmup_steps: int = 10,
    init_grad_scale: float = 1e32,
    clip_grad_norm: float = 1.0,
    grad_accumulation: int = 128,
    data_dir="data",
    model_dir="models/Llama-for-SpeechLM",
):
    model = LlamaForSpeechLM(LlamaForSpeechLMConfig(encoder_id=encoder_id, decoder_id=decoder_id)).cuda()

    encoder_processor = AutoProcessor.from_pretrained(encoder_id)
    decoder_processor = AutoProcessor.from_pretrained(decoder_id)
    decoder_processor.pad_token = decoder_processor.pad_token or decoder_processor.eos_token

    dataset = ConcatDataset(
        [
            torchaudio.datasets.LIBRISPEECH(root=data_dir, url="train-clean-100", download=True),
            torchaudio.datasets.LIBRISPEECH(root=data_dir, url="train-clean-360", download=True),
            torchaudio.datasets.LIBRISPEECH(root=data_dir, url="train-other-500", download=True),
            Clotho(root=data_dir, caption_idx=1),
            Clotho(root=data_dir, caption_idx=2),
            Clotho(root=data_dir, caption_idx=3),
            Clotho(root=data_dir, caption_idx=4),
            Clotho(root=data_dir, caption_idx=5),
        ]
    )

    def get_collate_fn(encoder_processor, decoder_processor):
        asr_prompt = """<|start_header_id|>user<|end_header_id|>

        Transcribe the audio.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        {}<|eot_id|>"""

        aac_prompt = """<|start_header_id|>user<|end_header_id|>

        Describe the audio.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        {}<|eot_id|>"""

        def collate_fn(
            batch: List[Tuple[torch.Tensor, int, str, int, int, int] | Tuple[torch.Tensor, int, str]],
        ) -> Dict[str, torch.Tensor]:
            """
            Args:
                batch: List of tuples.
                    ASR: (waveform, sample rate, transcript, speaker ID, chapter ID, utterance ID)
                    AAC: (waveform, sample rate, transcript)
            """

            encoder_inputs = encoder_processor(
                [item[0].squeeze(0).numpy() for item in batch],
                return_tensors="pt",
                return_attention_mask=True,
                sampling_rate=16000,
                device="cuda",
            ).to("cuda")

            decoder_inputs = decoder_processor(
                [
                    asr_prompt.format(item[2].lower()) if len(item) == 6 else aac_prompt.format(item[2].lower())
                    for item in batch
                ],
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            return {
                "input_features": encoder_inputs.input_features,
                "input_ids": decoder_inputs.input_ids,
                "encoder_attention_mask": encoder_inputs.attention_mask,
                "decoder_attention_mask": decoder_inputs.attention_mask,
            }

        return collate_fn

    loader = torch.utils.data.DataLoader(
        dataset, batch_size, True, collate_fn=get_collate_fn(encoder_processor, decoder_processor)
    )

    _train(
        model,
        loader,
        lr,
        epoch,
        warmup_steps,
        init_grad_scale,
        clip_grad_norm,
        grad_accumulation,
        model_dir,
    )


def generate_data(
    model_id="ryota-komatsu/Llama-for-SpeechLM",
    tts_id="kakao-enterprise/vits-vctk",
):
    model = LlamaForSpeechLM.from_pretrained(model_id).cuda()

    encoder_processor = AutoProcessor.from_pretrained(model.config.encoder_id)
    decoder_processor = AutoProcessor.from_pretrained(model.config.decoder_id)
    decoder_processor.pad_token = decoder_processor.pad_token or decoder_processor.eos_token

    tts_model = AutoModel.from_pretrained(tts_id).cuda()
    tts_tokenizer = AutoTokenizer.from_pretrained(tts_id)

    def filter_by_input(example):
        pattern = "[A-Za-z,.'!? ]+"
        noinput_pattern = r"no\s*input\s*(required)?\.?"
        return (
            example["input"] != ""
            and example["input"] != "Mon cheval est blanc"
            and example["input"] != "The bakery that I visited yesterday had freshly made croissants."
            and example["input"] != "Croissants are French pastries. The sky is blue."
            and not re.match(noinput_pattern, example["input"], re.IGNORECASE)
            and re.fullmatch(pattern, example["input"]) is not None
        )

    @torch.inference_mode()
    def add_audio(example):
        inputs = tts_tokenizer(example["input"], return_tensors="pt").to("cuda")
        output = tts_model(**inputs).waveform
        output = torchaudio.functional.resample(output, tts_model.config.sampling_rate, 16000)
        output = output.squeeze(0).cpu().numpy()
        return {"audio": {"array": output, "sampling_rate": 16000}}

    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.filter(filter_by_input)
    dataset = dataset.map(add_audio)
    dataset.push_to_hub("spoken-alpaca")


def finetune(
    model_id="ryota-komatsu/Llama-for-SpeechLM",
    dataset_id="ryota-komatsu/spoken-alpaca",
    model_dir="models/Llama-for-SpeechLM-Instruct",
    batch_size: int = 4,
    lr: float = 1e-3,
    epoch: int = 5,
    warmup_steps: int = 10,
    init_grad_scale: float = 1e32,
    clip_grad_norm: float = 1.0,
    grad_accumulation: int = 128,
):
    model = LlamaForSpeechLM.from_pretrained(model_id).cuda()

    encoder_processor = AutoProcessor.from_pretrained(model.config.encoder_id)
    decoder_processor = AutoProcessor.from_pretrained(model.config.decoder_id)
    decoder_processor.pad_token = decoder_processor.pad_token or decoder_processor.eos_token

    def is_train_example(example):
        return (
            len(example["audio"]["array"]) < 16000 * 30
            and len(example["instruction"]) < 102
            and len(example["output"]) < 838
        )

    dataset = load_dataset(dataset_id, split="train")
    dataset = dataset.with_format("torch")
    dataset = dataset.filter(is_train_example)

    def get_collate_fn(encoder_processor, decoder_processor):
        prompt = """<|start_header_id|>user<|end_header_id|>

        Below is an instruction that describes a task, paired with an audio input that provides further context. Transcribe the audio clip into English, and then write a response that appropriately completes the request.

        ### Instruction:
        {}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        ### Transcript:
        {}

        ### Response:
        {}<|eot_id|>"""

        def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            """
            Args:
                batch: List of the following example:
                    {
                        "instruction": "",
                        "input": "",
                        "output": "",
                        "text": "",
                        "audio": {"path": None, "array": tensor([...]), "sampling_rate": tensor(16000)},
                    }
            """

            encoder_inputs = encoder_processor(
                [item["audio"]["array"].numpy() for item in batch],
                return_tensors="pt",
                return_attention_mask=True,
                sampling_rate=16000,
                device="cuda",
            ).to("cuda")

            decoder_inputs = decoder_processor(
                [prompt.format(item["instruction"], item["input"], item["output"]) for item in batch],
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            return {
                "input_features": encoder_inputs.input_features,
                "input_ids": decoder_inputs.input_ids,
                "encoder_attention_mask": encoder_inputs.attention_mask,
                "decoder_attention_mask": decoder_inputs.attention_mask,
            }

        return collate_fn

    loader = torch.utils.data.DataLoader(
        dataset, batch_size, True, collate_fn=get_collate_fn(encoder_processor, decoder_processor)
    )

    _train(
        model,
        loader,
        lr,
        epoch,
        warmup_steps,
        init_grad_scale,
        clip_grad_norm,
        grad_accumulation,
        model_dir,
    )


def eval(
    encoder_id="openai/whisper-small.en",
    decoder_id="meta-llama/Llama-3.2-1B-Instruct",
    dataset_id="ryota-komatsu/spoken-alpaca",
    model_dir="models/Llama-for-SpeechLM-Instruct",
    max_length: int = 1024,
    do_sample: bool = False,
    num_beams: int = 5,
):
    model = LlamaForSpeechLM.from_pretrained(model_dir).cuda()

    encoder_processor = AutoProcessor.from_pretrained(encoder_id)
    decoder_processor = AutoProcessor.from_pretrained(decoder_id)
    decoder_processor.pad_token = decoder_processor.pad_token or decoder_processor.eos_token

    prompt = """<|start_header_id|>user<|end_header_id|>

    Below is an instruction that describes a task, paired with an audio input that provides further context. Transcribe the audio clip into English, and then write a response that appropriately completes the request.

    ### Instruction:
    {}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    """

    def is_test_example(example):
        return (
            len(example["audio"]["array"]) < 16000 * 30
            and 102 <= len(example["instruction"])
            and len(example["output"]) < 838
        )

    dataset = load_dataset(dataset_id, split="train")
    dataset = dataset.with_format("torch")
    dataset = dataset.filter(is_test_example)

    loader = torch.utils.data.DataLoader(dataset, shuffle=True)

    for item in loader:
        encoder_inputs = encoder_processor(
            item["audio"]["array"].numpy(),
            return_tensors="pt",
            return_attention_mask=True,
            sampling_rate=16000,
            device="cuda",
        ).to("cuda")

        decoder_inputs = decoder_processor(
            prompt.format(item["instruction"][0]),
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        generated_ids = model.generate(
            encoder_inputs.input_features,
            decoder_inputs.input_ids,
            encoder_attention_mask=encoder_inputs.attention_mask,
            decoder_attention_mask=decoder_inputs.attention_mask,
            max_length=max_length,
            do_sample=do_sample,
            num_beams=num_beams,
        )
        generated_txt = decoder_processor.batch_decode(generated_ids, skip_special_tokens=True)


if __name__ == "__main__":
    eval()
