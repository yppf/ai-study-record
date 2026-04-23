import pandas as pd
import torch


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    section("1. Conditional probability intuition")
    total = 10
    rain_and_late = 3
    rain = 4
    p_rain = rain / total
    p_late_given_rain = rain_and_late / rain
    print("P(rain) =", p_rain)
    print("P(late | rain) =", p_late_given_rain)

    section("2. Expectation and variance")
    values = torch.tensor([0.0, 1.0, 2.0, 3.0])
    probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
    expectation = (values * probs).sum()
    variance = (((values - expectation) ** 2) * probs).sum()
    print("values =", values)
    print("probs =", probs)
    print("expectation =", expectation.item())
    print("variance =", variance.item())

    section("3. Tiny pandas preprocessing")
    data = pd.DataFrame(
        {
            "age": [25, None, 31, 28],
            "city": ["shanghai", "beijing", None, "shenzhen"],
            "label": [1, 0, 1, 0],
        }
    )
    print("raw data:\n", data)

    features = data[["age", "city"]].copy()
    features["age"] = features["age"].fillna(features["age"].mean())
    features = pd.get_dummies(features, dummy_na=True)
    print("processed features:\n", features)
    print("feature shape =", features.shape)

    section("4. What to try next")
    print("Try changing the probabilities so they still sum to 1.")
    print("Try adding one more city and one more missing value.")


if __name__ == "__main__":
    main()
