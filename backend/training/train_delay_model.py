"""Train the delay-risk model and print how it compares with the naive ETA rule.

python -m training.train_delay_model
"""

import json

from app.tools.delay_model import MODEL_PATH, train

if __name__ == "__main__":
    report = train()
    print(json.dumps({"saved_to": str(MODEL_PATH), **report}, indent=2))
