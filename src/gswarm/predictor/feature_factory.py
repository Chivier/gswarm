from typing import Dict, Any, List
from pydantic import BaseModel
from enum import Enum

all_features = [
    # LLM
    "input_token_len",
    "top_p",
    "temperature",
    # Diffusion Text2Img
    "prompt_token_len",
    # Diffusion Img2Img & OCR
    "input_image_pixel_num",
    "input_image_height",
    "input_image_width",
    "input_image_channel_num",
    # Diffusion common
    "output_image_height",
    "output_image_width",
    "num_inference_steps",
    # Common features
    "batch_size",
    "model_size",
    "inference_time",
]

supported_model_features = {
    "llm": ["input_token_len", "top_p", "temperature"],
    "diffusion_text2img": ["prompt_token_len"],
    "diffusion_img2img": [
        "input_image_pixel_num",
        "input_image_height",
        "input_image_width",
        "input_image_channel_num",
        "output_image_height",
        "output_image_width",
    ],
    "diffusion_ocr": [
        "input_image_pixel_num",
        "input_image_height",
        "input_image_width",
        "input_image_channel_num",
        "output_image_height",
        "output_image_width",
    ],
    "all": [
        "batch_size",
        "model_size",
        "inference_time",
    ],
}


class ModelFeatures(BaseModel):
    model_name: str
    device_name: str
    feature: List[Any]


class ModelType(str, Enum):
    LLM = "llm"
    DIFFUSION_TEXT2IMG = "diffusion_text2img"
    DIFFUSION_IMG2IMG = "diffusion_img2img"
    DIFFUSION_OCR = "diffusion_ocr"


def encode_model_name(model_name: str):
    pass


def encode_device_name(device_name: str):
    pass


def feature_to_list(feature: ModelFeatures) -> List[Any]:
    """
    Convert ModelFeatures to a list of features.

    Args:
        feature (ModelFeatures): ModelFeatures object.

    Returns:
        List[Any]: List of features.
    """
    model_name_feature = encode_model_name(feature.model_name)
    device_name_feature = encode_device_name(feature.device_name)
    feature_list = feature.features if feature.features else []
    feature_list.extend(model_name_feature)
    feature_list.extend(device_name_feature)
    return feature_list


def feature_factory(
    model_name: str, device_name: str, model_type: ModelType, features: Dict[str, Any]
) -> ModelFeatures:
    """
    Factory function to create features for LLM models.

    Args:
        model_name (str): Name of the model.
        device_name (str): Name of the device.
        features (Dict[str, Any]): Features to be processed.

    Returns:
        Dict[str, Any]: Processed features.
    """
    # Check if all required features are present
    required_features = supported_model_features.get(model_type, [])
    required_features.extend(supported_model_features["all"])
    for feature in required_features:
        if feature not in features:
            raise ValueError(
                f"Missing required feature: {feature}, for model: {model_name}. Required features are: {required_features}"
            )

    feature = ModelFeatures(
        model_name=model_name, device_name=device_name, features=[features.get(name, 0) for name in all_features]
    )

    # Convert all feature to a list, and store back to feature object
    feature.feature = feature_to_list(feature)

    return feature
