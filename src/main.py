import os

print("Step 1: Training Custom CNN")

os.system("python train_custom_cnn.py")

print("Step 2: Evaluating Model")

os.system("python evaluate_model.py")

print("Step 3: Training Transfer Learning Models")

os.system("python transfer_learning_models.py")

print("Step 4: Comparing Models")

os.system("python model_comparison.py")

print("Step 5: Running Inference")

os.system("python inference_pipeline.py")

print("Pipeline completed successfully")