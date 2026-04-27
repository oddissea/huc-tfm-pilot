# src/encoders/encoder_factory.py

from typing import Union
from .base_encoder import BaseEncoder
from .dual_stream_encoder import DualStreamEncoder
from .resnet_encoder import load_resnet_encoder
from .bit_encoder import load_bit_encoder
from ..config import config


def create_encoder(
        architecture: str = None,
        patch_mode: str = None,
        fusion_mode: str = None
) -> Union[BaseEncoder, DualStreamEncoder]:
    """
    Factory function para crear encoder según configuración.

    Args:
        architecture: 'resnet18'|'resnet50'|'bitm' (usa config si None)
        patch_mode: 'original'|'dual' (usa config si None)
        fusion_mode: 'concat'|'add'|'attention' (usa config si None)

    Returns:
        Encoder configurado (BaseEncoder o DualStreamEncoder)
    """
    arch = architecture or config.MODEL_TYPE
    mode = patch_mode or config.PATCH_MODE
    fusion = fusion_mode or config.FUSION_MODE

    # Crear encoder base
    if arch in ['resnet18', 'resnet50']:
        base = load_resnet_encoder(arch, pretrained=True)
    elif arch == 'bitm':
        base = load_bit_encoder(pretrained=True)
    else:
        raise ValueError(f"Arquitectura desconocida: {arch}")

    # Envolver en dual-stream si es necesario
    if mode == 'dual':
        return DualStreamEncoder(
            base_encoder=base,
            fusion_mode=fusion,
            freeze_backbone=config.FREEZE_ENCODER
        )
    else:
        return base