"""Shared normalization and rotary position operations."""

import torch


def rms_norm(hidden, weight, epsilon):
    value = hidden.float()
    normalized = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + epsilon)
    return normalized.to(hidden.dtype) * weight


def rotary(query, key, positions, inv_freq, attention_factor=1.0):
    frequency = positions.float()[:, None] * inv_freq[None, :]
    phase = torch.cat((frequency, frequency), -1)
    cosine = (phase.cos() * attention_factor).to(query.dtype)[:, None, :]
    sine = (phase.sin() * attention_factor).to(query.dtype)[:, None, :]

    def rotate(value):
        left, right = value.chunk(2, -1)
        return value * cosine + torch.cat((-right, left), -1) * sine

    return rotate(query), rotate(key)
