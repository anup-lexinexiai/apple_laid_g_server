#!/usr/bin/env python3
"""Test script for LLM implementations"""

import os
os.environ['ENVIRONMENT'] = 'local'

from llms.llm_factory import create_llm
import json

def test_basic_chat():
    """Test basic chat without functions"""
    print("\n=== Testing Basic Chat ===")
    
    models = ['llmopenaigpt', 'llmclaudehaiku35', 'llmgemini']
    
    for model_name in models:
        print(f"\nTesting {model_name}...")
        try:
            llm = create_llm(model_name)
            
            complete_chat = {
                'instruction': 'You are a helpful assistant.',
                'conversation_history': [
                    {'role': 'user', 'content': 'Say hello in 5 words'}
                ]
            }
            
            response = llm.llm_response(complete_chat)
            print(f"Response: {response.get('message', 'No message')}")
            print(f"Cost: {llm.get_cost()}")
            
        except Exception as e:
            print(f"Error: {e}")

def test_function_calling():
    """Test function calling"""
    print("\n=== Testing Function Calling ===")
    
    llm = create_llm('llmopenaigpt')
    
    complete_chat = {
        'instruction': 'You are a helpful assistant that can get weather.',
        'conversation_history': [
            {'role': 'user', 'content': 'What is the weather in NYC?'}
        ],
        'available_functions': [
            {
                'name': 'get_weather',
                'description': 'Get current weather for a location',
                'parameters': {
                    'location': {
                        'type': 'string',
                        'description': 'City name',
                        'required': True
                    }
                }
            }
        ]
    }
    
    response = llm.llm_response(complete_chat)
    print(f"Response: {json.dumps(response, indent=2)}")

def test_function_with_output():
    """Test function calling with output"""
    print("\n=== Testing Function Output ===")
    
    llm = create_llm('llmclaudehaiku35')
    
    # Simulate conversation with function call and output
    complete_chat = {
        'conversation_history': [
            {'role': 'user', 'content': 'What is the weather in NYC?'},
            {
                'type': 'function_call',
                'name': 'get_weather',
                'arguments': '{"location": "NYC"}',
                'call_id': 'call_123'
            },
            {
                'type': 'function_call_output',
                'name': 'get_weather',
                'output': 'The weather in NYC is 72°F and sunny.',
                'call_id': 'call_123'
            }
        ]
    }
    
    response = llm.llm_response(complete_chat)
    print(f"Response: {response.get('message', 'No message')}")

if __name__ == '__main__':
    test_basic_chat()
    test_function_calling()
    test_function_with_output()