from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/endpoint_1")
async def endpoint_1():
    return {"message": "FastAPI Endpoint 1 Active!"}

@app.get("/endpoint_2")
async def endpoint_2():
    return {"message": "FastAPI Endpoint 2 Active!"}

# Create a Bedrock Runtime client in the AWS Region you want to use.
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Set the model ID, e.g., Titan Text Premier.
model_id = "anthropic.claude-3-haiku-20240307-v1:0"

# Input model for the conversation
class ConversationRequest(BaseModel):
    user_message: str

@app.post("/endpoint_3/")
async def converse(request: ConversationRequest):
    # Use the user message directly from the request
    user_message = request.user_message
    conversation = [
        {
            "role": "user",
            "content": [{"text": user_message}],
        }
    ]

    try:
        # Send the message to the model, using a basic inference configuration.
        response = client.converse(
            modelId=model_id,
            messages=conversation,
            inferenceConfig={"maxTokens": 4096, "temperature": 0},
            additionalModelRequestFields={"top_k": 250}
        )

        # Extract and return the response text.
        response_text = response["output"]["message"]["content"][0]["text"]
        print(response_text)
        return {"model_output": response_text}

    except (ClientError, Exception) as e:
        raise HTTPException(status_code=500, detail=f"ERROR: Can't invoke '{model_id}'. Reason: {e}")