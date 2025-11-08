import os, torch
from flask import Flask
from flask_cors import CORS  # Add this
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

def create_app():
    """
    Create and configure the Flask application.
    """
    app = Flask(__name__)
    CORS(app)  # Add this - enables CORS for all routes
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-secret-key-change-me')

    # Load model and tokenizer
    daimon_model_name = "meta-llama/Llama-3.2-3B-Instruct"
    app.untrained_model = AutoModelForCausalLM.from_pretrained(
        daimon_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token="", 
    )
    app.tokenizer = AutoTokenizer.from_pretrained(daimon_model_name)
    
    app.trained_model = AutoModelForCausalLM.from_pretrained(
        './checkpoints/step_74500_epoch_5',
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token="", 
    )

    # Set pad token if not set
    app.tokenizer.pad_token = app.tokenizer.eos_token

    avvocato_model_name = "Equall/Saul-7B-Instruct-v1"
    
    app.avvocato_tokenizer = AutoTokenizer.from_pretrained(avvocato_model_name)
    app.avvocato_model = AutoModelForCausalLM.from_pretrained(
        avvocato_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    return app
