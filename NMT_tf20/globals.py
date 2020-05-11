import os

PATH_SERVER = "/project/"
TEST_PATH_SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)))
TEST_MODEL_NAME = "Transformer_OneHot"
TEST_RUN_NAME = "best-model-transformer"
TRAINING = False
print(f"Globals: TRAINING MODE : {TRAINING}")
