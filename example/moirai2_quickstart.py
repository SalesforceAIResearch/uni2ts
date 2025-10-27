import numpy as np
from uni2ts.model.moirai2 import Moirai2Module, Moirai2Forecast

# Load pretrained Moirai2 weights from Hugging Face
module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")

# Build the forecasting wrapper
forecast = Moirai2Forecast(
    module=module,
    prediction_length=100,     # forecast horizon
    context_length=1680,       # how many past points the model looks at
    target_dim=1,              # univariate
    feat_dynamic_real_dim=0,   # no known future covariates
    past_feat_dynamic_real_dim=0,  # no past-only covariates
)

# Example past series (replace with your own 1D numpy array)
y = np.sin(np.linspace(0, 50, 1200)).astype(np.float32)

# Run prediction; input is a list of series
pred = forecast.predict([y])  # shape: (batch=1, num_quantiles, future_time)

# Take median forecast (q=0.5)
q_levels = list(forecast.module.quantile_levels)
median_idx = q_levels.index(0.5)
yhat = pred[0, median_idx]  # shape: (prediction_length,)
print("sample past data: --------------------------------")
print(y)
print("sample forecast data: --------------------------------")
print(yhat)