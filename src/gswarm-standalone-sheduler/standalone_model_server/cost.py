from llm_cost_model import LLMCostModel
from sd_cost_model import SDCostModel
import random

sd_cost_model = SDCostModel()
llm_cost_model = LLMCostModel()


def get_estimation_cost(model_type: str, model_name: str, device: str, data_features) -> float:
    """
    Get the estimation cost of a model for given data features.
    For LLM models:
    {"prompt_length": int}
    or
    {"prompt": str}
    For Diffusion models:
    {"height": int, "width": int}
    """
    result = []
    for current_feature in data_features:
        if model_type == "llm":
            if "prompt_length" in current_feature:
                result.append(llm_cost_model.predict(current_feature["prompt_length"]))
            elif "prompt" in current_feature:
                result.append(llm_cost_model.predict_str(current_feature["prompt"]))
            else:
                raise ValueError("Invalid data features for LLM model prediction.")
        elif model_type == "diffusion":
            result.append(
                sd_cost_model.predict(model_name, device, current_feature["height"], current_feature["width"])
            )
        else:
            result.append(random.uniform(1.0, 100))  # Return a random cost for unknown models
    return result


def update_predictor(model_type: str, model_name: str, device: str, data_features) -> None:
    """
    Update the predictor model with new data features.
    For LLM models:
    {"prompt_length": int, "output_length": int}
    or
    {"prompt": str, "output": str}
    For Diffusion models:
    {"height": int, "width": int, "processing_time": float}
    """
    for current_feature in data_features:
        if model_type == "llm":
            if "prompt_length" not in current_feature and "prompt" in current_feature and "output" in current_feature:
                llm_cost_model.update_model(current_feature["prompt"], current_feature["output"])
            elif "prompt_length" in current_feature and "output_length" in current_feature:
                llm_cost_model.update_model(current_feature["prompt_length"], current_feature["output_length"])
            else:
                raise ValueError("Invalid data features for LLM model update.")
        elif model_type == "diffusion":
            sd_cost_model.update_model(
                model_name,
                device,
                current_feature["height"],
                current_feature["width"],
                current_feature["processing_time"],
            )
        else:
            pass


if __name__ == "__main__":
    # Example usage
    model_name = "llm-gpt-4o"
    device = "NVIDIAH20"
    data_features = [{"prompt_length": 100}, {"prompt": "Hello, world!"}]
    cost = get_estimation_cost("llm", model_name, device, data_features)
    print(f"Estimated cost for {model_name} on {device}: {cost}")
    model_name = "sd-stable-diffusion"
    device = "NVIDIAH20"
    base_number = 0.5
    base_pixel = 128 * 128
    height = [128, 256, 512]
    width = [128, 256, 512]
    for h in height:
        for w in width:
            update_predictor(
                "diffusion",
                model_name,
                device,
                [{"height": h, "width": w, "processing_time": base_number * (h * w / base_pixel)}],
            )

    data_features = [1024, 1024]  # Example features
    cost = get_estimation_cost("diffusion", model_name, device, [{"height": 1024, "width": 1024}])
    print(f"Estimated cost for {model_name} on {device}: {cost}")
