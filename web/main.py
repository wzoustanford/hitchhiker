from flask import request, jsonify
from web import create_app
import torch

app = create_app()

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Chat endpoint - receives POST requests with JSON payload.
    Expected payload: {"message": "user message here"}
    """
    try:
        # Extract JSON payload from request
        request_payload = request.get_json(force=True, silent=True)
        if not request_payload:
            return jsonify({"error": "Invalid JSON request"}), 400

        user_message = request_payload.get('message')
        if not user_message:
            return jsonify({"error": "No message provided"}), 400


        torch.set_grad_enabled(False)
        messages = [
            {"role": "system", "content": "You are a helpful financial assistant and an expert with US equities."},
            {"role": "user", "content": user_message}
        ]

        # Apply chat template
        prompt = app.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize and generate
        inputs = app.tokenizer(prompt, return_tensors="pt").to(app.untrained_model.device)
        """
        with torch.inference_mode():  # Add this context manager
            outputs = app.untrained_model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1,
            )

        response = app.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's response
        untrained_assistant_response = response.split("assistant")[-1].strip()
        """
        
        with torch.inference_mode():  # Add this context manager
            outputs = app.trained_model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1,
            )

        response = app.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's response
        trained_assistant_response = response.split("assistant")[-1].strip()

        # TODO: Add your chat processing logic here
        # For now, just echo back the message
        response_message = trained_assistant_response 

        return jsonify({
            "message": response_message,
            "status": "success"
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/api/chat_avvocato', methods=['POST'])
def chat_avvocato():
    """
    Chat endpoint - receives POST requests with JSON payload.
    Expected payload: {"message": "user message here"}
    """
    try:
        # Extract JSON payload from request
        request_payload = request.get_json(force=True, silent=True)
        if not request_payload:
            return jsonify({"error": "Invalid JSON request"}), 400

        user_message = request_payload.get('message')
        if not user_message:
            return jsonify({"error": "No message provided"}), 400


        torch.set_grad_enabled(False)
        messages = [
            {"role": "user", "content": user_message}
        ]

        # Apply chat template
        prompt = app.avvocato_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize and generate
        inputs = app.avvocato_tokenizer(prompt, return_tensors="pt").to(app.untrained_model.device)
        """
        with torch.inference_mode():  # Add this context manager
            outputs = app.untrained_model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1,
            )

        response = app.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's response
        untrained_assistant_response = response.split("assistant")[-1].strip()
        """
        
        with torch.inference_mode():  # Add this context manager
            outputs = app.avvocato_model.generate(
                **inputs,
                max_new_tokens=1024,
                #temperature=0.7,
                #do_sample=True,
                #top_p=0.95,
                #top_k=50,
                #repetition_penalty=1.1,
            )

        response = app.avvocato_tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        #print(app.avvocato_tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
        # Extract only the assistant's response
        response_message = response.split("assistant")[-1].strip()
        
        return jsonify({
            "message": response_message,
            "status": "success"
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
