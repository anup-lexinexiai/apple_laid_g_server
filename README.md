# apple_laid_g_server
previously server in aws , we make it lean and bring to GCP

pip install -r requirements.txt


cp configs/config.example.yaml configs/config.local.yaml
# Edit config.local.yaml with your API keys

run locally
python main.py



test api

# Get credits
curl -H "Authorization: Bearer test-token" http://localhost:8080/api/get_user_credit

# Call LLM
curl -X POST http://localhost:8080/api/get_llm_response \
  -H "Authorization: Bearer test-token" \
  -H "Content-Type: application/json" \
  -d '{
    "llm_class_name": "llmopenaigpt",
    "complete_chat": {
      "conversation_history": [
        {"role": "user", "content": "Hello!"}
      ]
    }
  }'
