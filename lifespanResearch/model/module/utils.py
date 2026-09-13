import torch
from pathlib import Path
from esm.models.esmc import ESMC
from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    ESMProteinError,
    LogitsConfig,
    LogitsOutput,
    ProteinType,
)
from esm.tokenization import get_esm3_model_tokenizers, get_esmc_model_tokenizers
from esm.utils.constants.esm3 import data_root


def load_local_esmc(device: str | torch.device = "cpu", use_flash_attn: bool = True, root: str | Path = "esmc-600m-2024-12"):
    """Load a local ESMC model from the specified path.
    Args:
        model_path (str): Path to the local model directory.
        use_flash_attn (bool, optional): Whether to use Flash Attention. Defaults to True.

    Returns:
        ESMC: The loaded ESMC model.

    Note:what is the function of '|' ? '|' is used for type hinting to indicate that the parameter can be of multiple types. 
    """
    root = Path(root) # Convert root to Path object

    # 1、 Load tokenizer
    tokenizer = get_esmc_model_tokenizers()

    # 2、 Load model
    model = ESMC(
        d_model=1152,
        n_heads=18,
        n_layers=36,
        tokenizer=tokenizer,
        use_flash_attn=use_flash_attn
    ).eval()

    # 3、 Load model weights
    weights_path = root / "data" / "weights" / "esmc_600m_2024_12_v0.pth"
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)

    return model,tokenizer

