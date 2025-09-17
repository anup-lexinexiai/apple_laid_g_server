@app.route('/api/get_llm_response', methods=['POST'])
@require_auth
def get_llm_response():
    user_id = request.user_id
    data = request.get_json()
    
    llm_model_name = data.get('llm_class_name')
    complete_chat = data.get('complete_chat', {})
    query_type = complete_chat.get('query_type', 'default')
    
    if not llm_model_name:
        return jsonify({'error': 'Missing llm_class_name'}), 400
    
    # Initialize services
    credit_service = CreditService()
    llm_router = LLMRouter()
    
    # Get model cost
    model_cost = llm_router.get_model_cost(llm_model_name)
    
    # Check credits for billable query types
    if query_type in ["agent_listening_from", "query_from_func_exec"]:
        if not credit_service.has_sufficient_credits(user_id, model_cost):
            return jsonify({
                'llm_response': "You don't have enough **Lexi Credit**, Please **Recharge Credits** under **More Options**",
                'function_call': None,
                'credit_remaining': credit_service.get_balance(user_id)
            }), 200
    
    try:
        # Process LLM request
        response = llm_router.process_request(llm_model_name, complete_chat)
        
        # Deduct credits if billable
        new_balance = credit_service.get_balance(user_id)
        if query_type in ["agent_listening_from", "query_from_func_exec"]:
            new_balance = credit_service.deduct_credits(
                user_id, 
                model_cost, 
                llm_model_name,
                query_type
            )
        
        return jsonify({
            'llm_response': response.get('message', ''),
            'function_call': response.get('function_call'),
            'credit_remaining': new_balance
        })
        
    except Exception as e:
        import traceback
        print(f"Error in LLM processing: {str(e)}")
        print(traceback.format_exc())
        
        return jsonify({
            'error': f'LLM processing error: {str(e)}',
            'credit_remaining': credit_service.get_balance(user_id)
        }), 500
        