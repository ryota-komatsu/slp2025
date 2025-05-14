from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchaudio
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
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
        self.linear = nn.Linear(encoder_hidden_size, decoder_hidden_size, bias=bias)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.pool(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.linear(hidden_states)
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

    def forward(
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

        encoder_hidden_states = self.adapter(encoder_hidden_states)

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

        labels = torch.cat(
            (
                torch.full(encoder_hidden_states.shape[:2], -100, dtype=input_ids.dtype, device=input_ids.device),
                input_ids,
            ),
            dim=1,
        )

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
        encoder_outputs = self.encoder(input_features)
        encoder_hidden_states = encoder_outputs[0]

        lengths = self.encoder._get_feat_extract_output_lengths(encoder_attention_mask.sum(dim=1, keepdim=True))
        lengths = lengths // self.config.adapter_kernel_size

        encoder_hidden_states = self.adapter(encoder_hidden_states)

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

        generated_ids = self.decoder.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)
        return generated_ids


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


def get_collate_fn(encoder_processor, decoder_processor):
    prompt = """
    <|start_header_id|>user<|end_header_id|>

    Transcribe the audio clip into English.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    {}<|eot_id|>
    """

    def collate_fn(batch: List[Tuple[torch.Tensor, int, str, int, int, int]]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: List of (waveform, sample rate, transcript, speaker ID, chapter ID, utterance ID)
        """

        encoder_inputs = encoder_processor(
            [item[0].squeeze(0).numpy() for item in batch],
            return_tensors="pt",
            return_attention_mask=True,
            sampling_rate=16000,
            device="cuda",
        ).to("cuda")

        decoder_inputs = decoder_processor(
            [prompt.format(item[2].lower()) for item in batch],
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


def train(
    encoder_id="openai/whisper-small.en",
    decoder_id="meta-llama/Llama-3.2-1B-Instruct",
    batch_size: int = 4,
    lr: float = 1e-3,
    epoch: int = 1,
    warmup_steps: int = 10,
    clip_grad_norm: float = 1.0,
    grad_accumulation: int = 128,
    data_dir="data",
    model_dir="models/llama-for-speech-lm",
):
    model = LlamaForSpeechLM(LlamaForSpeechLMConfig(encoder_id=encoder_id, decoder_id=decoder_id)).cuda()

    encoder_processor = AutoProcessor.from_pretrained(encoder_id)
    decoder_processor = AutoProcessor.from_pretrained(decoder_id)
    decoder_processor.pad_token = decoder_processor.pad_token or decoder_processor.eos_token

    trainset = torchaudio.datasets.LIBRISPEECH(root=data_dir, url="train-clean-100", download=True)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size, True, collate_fn=get_collate_fn(encoder_processor, decoder_processor)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # learning rate scheduler
    lr_scheduler = get_lr_schedule(
        optimizer,
        len(train_loader) // grad_accumulation * epoch,
        warmup_steps,
        lr,
        lr * 0.1,
    )

    scaler = torch.amp.GradScaler("cuda")
    writer = SummaryWriter()

    step = 0

    for epoch in range(1, epoch + 1):
        model.train()

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch}")):
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


def eval(
    encoder_id="openai/whisper-small.en",
    decoder_id="meta-llama/Llama-3.2-1B-Instruct",
    data_dir="data",
    model_dir="models/llama-for-speech-lm",
    max_length: int = 1024,
):
    model = LlamaForSpeechLM.from_pretrained(model_dir).cuda()

    encoder_processor = AutoProcessor.from_pretrained(encoder_id)
    decoder_processor = AutoProcessor.from_pretrained(decoder_id)
    decoder_processor.pad_token = decoder_processor.pad_token or decoder_processor.eos_token

    prompt = """
    <|start_header_id|>user<|end_header_id|>

    Transcribe the audio clip into English.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    """

    testset = torchaudio.datasets.LIBRISPEECH(root=data_dir, url="test-clean", download=True)
    test_loader = torch.utils.data.DataLoader(testset)

    for item in test_loader:
        encoder_inputs = encoder_processor(
            item[0].squeeze(0).numpy(),
            return_tensors="pt",
            return_attention_mask=True,
            sampling_rate=16000,
            device="cuda",
        ).to("cuda")

        decoder_inputs = decoder_processor(
            prompt,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        generated_ids = model.generate(
            encoder_inputs.input_features,
            decoder_inputs.input_ids,
            encoder_attention_mask=encoder_inputs.attention_mask,
            decoder_attention_mask=decoder_inputs.attention_mask,
            max_length=max_length,
        )
        generated_txt = decoder_processor.batch_decode(generated_ids)


if __name__ == "__main__":
    train()
