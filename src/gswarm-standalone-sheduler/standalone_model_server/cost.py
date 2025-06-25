from llm_cost_model import LLMCostModel
from sd_cost_model import SDCostModel
import random

sd_cost_model = SDCostModel()
llm_cost_model = LLMCostModel()


def get_estimation_cost(model_type: str, model_name: str, device: str, *data_features) -> float:
    """
    Get the estimation cost of a model for given data features.
    """
    if model_type == "llm":
        model_name = model_name[4:]  # Remove 'llm-' prefix
        return llm_cost_model.predict(*data_features)
    elif model_type == "diffusion":
        model_name = model_name[3:]  # Remove 'sd' prefix
        if len(data_features) < 3:
            return sd_cost_model.predict(model_name, device, *data_features)
        else:
            sd_cost_model.update_model(model_name, device, *data_features)
    else:
        return random.uniform(1.0, 100)  # Return a random cost for unknown models


if __name__ == "__main__":
    # Example usage
    model_name = "llm-gpt-4o"
    device = "NVIDIAH20"
    data_features = ["Quick fox jump over lazy dog"]  # Example features like height and width
    cost = get_estimation_cost(model_name, device, *data_features)
    print(f"Estimated cost for {model_name} on {device}: {cost}")
    model_name = "sd-stable-diffusion"
    device = "NVIDIAH20"
    base_number = 0.5
    base_pixel = 128 * 128
    height = [128, 256, 512]
    width = [128, 256, 512]
    for h in height:
        for w in width:
            get_estimation_cost(model_name, device, h, w, base_number * (h * w / base_pixel))

    data_features = [1024, 1024]  # Example features
    cost = get_estimation_cost(model_name, device, *data_features)
    print(f"Estimated cost for {model_name} on {device}: {cost}")
