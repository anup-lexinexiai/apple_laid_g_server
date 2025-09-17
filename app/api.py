from flask import Blueprint, request, jsonify
from app.auth import require_auth
from services.credit_service import CreditService
from services.llm_router import LLMRouter

# Create a Blueprint
api_bp = Blueprint('api', __name__)

def register_routes(app):
    """Register API routes with the Flask app."""
    
    @app.route('/api/get_user_credit', methods=['GET'])
    @require_auth
    def get_user_credit():
        user_id = request.user_id
        credit_service = CreditService()
        balance = credit_service.get_balance(user_id)
        return jsonify({'lexi_credit': balance})

    @app.route('/api/get_llm_response', methods=['POST'])
    @require_auth
    def get_llm_response():
        user_id = request.user_id
        data = request.get_json()
        
        llm_model_name = data.get('llm_class_name')
        complete_chat = data.get('complete_chat', {})
        
        if not llm_model_name or not complete_chat:
            return jsonify({'error': 'Missing llm_class_name or complete_chat'}), 400
        
        credit_service = CreditService()
        llm_router = LLMRouter()
        
        # Get model cost and check credits
        model_cost = llm_router.get_model_cost(llm_model_name)
        if not credit_service.has_sufficient_credits(user_id, model_cost):
            return jsonify({
                'llm_response': "You don't have enough **Lexi Credit**, Please **Recharge Credits** under **More Options**",
                'function_call': None,
                'credit_remaining': credit_service.get_balance(user_id)
            }), 200
        
        try:
            # Process LLM request
            response = llm_router.process_request(llm_model_name, complete_chat)
            
            # Deduct credits
            new_balance = credit_service.deduct_credits(user_id, model_cost, llm_model_name)
            
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

    @app.route('/api/recharge_user_credit', methods=['POST'])
    @require_auth
    def recharge_user_credit():
        user_id = request.user_id
        data = request.get_json()
        credits_to_add = data.get('lexi_credits_added')

        if credits_to_add is None:
            return jsonify({'error': 'Missing lexi_credits_added parameter'}), 400
        
        try:
            credits_to_add = float(credits_to_add)
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid value for lexi_credits_added'}), 400
            
        credit_service = CreditService()
        new_balance = credit_service.add_credits(user_id, credits_to_add)
        
        return jsonify({'status': 'success', 'new_lexi_credit': new_balance})
