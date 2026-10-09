"""원본 cell 10의 모델. 수치 연산은 보존했으며 docs/REPRODUCIBILITY.md를 참고하세요."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper


class IntegratedWhisperFFT(nn.Module):
    def __init__(self, num_classes, whisper_size="small"):
        super().__init__()

        whisper_full = whisper.load_model(whisper_size)
        self.encoder = whisper_full.encoder

        for p in self.encoder.parameters():
            p.requires_grad = True

        d_model = self.encoder.conv2.out_channels  # small=768
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        self.n_fft = 400
        self.hop = 160
        self.mel_bins = 80
        self.mel_len = 300

        filters = torch.tensor(
            whisper.audio.mel_filters(torch.device("cpu"), self.mel_bins),
            dtype=torch.float32
        )
        self.register_buffer("mel_filters", filters)
        self.register_buffer("window", torch.hann_window(self.n_fft))

        self.embedding_len = 150
        pe = self.encoder.positional_embedding
        self.encoder.positional_embedding = nn.Parameter(pe[:self.embedding_len, :])

    def audio_to_logmel(self, audio):
        """
        audio: [B,48000] 또는 [48000]
        return: [B,80,300]
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # [1,T]

        # ✅ Whisper 방식 reflect pad (200,200) : pad는 (N,C,L)에서 안전
        x = audio.unsqueeze(1)                 # [B,1,T]
        x = F.pad(x, (200, 200), mode="reflect")
        x = x.squeeze(1)                       # [B,T+400]

        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window.to(x.device),
            return_complex=True
        )  # [B, F, Frames]

        mag = stft.abs().pow(2)  # power

        mel_filters = self.mel_filters.to(x.device)     # [80, F]
        mel = torch.matmul(mel_filters, mag)            # [B,80,Frames]

        logmel = torch.clamp(mel, min=1e-10).log10()
        logmel = torch.maximum(logmel, logmel.max() - 8.0)
        logmel = (logmel + 4.0) / 4.0

        # 길이 300 고정
        T = logmel.size(-1)
        if T > self.mel_len:
            logmel = logmel[..., :self.mel_len]
        elif T < self.mel_len:
            logmel = F.pad(logmel, (0, self.mel_len - T))

        return logmel

    def forward_features(self, audio):
        logmel = self.audio_to_logmel(audio)   # [B,80,300]
        enc = self.encoder(logmel)             # [B,150,D]
        emb = enc.mean(dim=1)                  # [B,D]
        return emb

    def forward(self, audio):
        emb = self.forward_features(audio)
        logits = self.classifier(emb)
        return logits
