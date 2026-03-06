"""
AnuRAG: Agent Module
ReAct-style agent for multimodal reasoning about analog circuits
Supports Gemini and Claude via the LLM provider abstraction.
"""

import os
import re
import json
import shutil
import time
from typing import Dict, Any, Optional, List, Tuple
from dotenv import load_dotenv

load_dotenv()

# Import centralized configuration
from config import (
    GEMINI_CHAT_MODEL,
    TEMPERATURE,
    MAX_OUTPUT_TOKENS,
    get_model_costs
)
from llm_provider import get_llm_provider

from messages import system_message
from search import main as search_db
from load_titles import load_titles

# Cost estimation -- use config-driven costs
def calculate_cost(input_tokens: int, output_tokens: int) -> Dict[str, float]:
    """Calculate estimated cost based on token usage and active model."""
    costs = get_model_costs()
    input_cost = (input_tokens / 1_000_000) * costs.get('input', 0.10)
    output_cost = (output_tokens / 1_000_000) * costs.get('output', 0.40)
    total_cost = input_cost + output_cost
    return {
        'input_cost': input_cost,
        'output_cost': output_cost,
        'total_cost': total_cost,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens
    }


class GeminiAgent:
    """
    ReAct-style agent using the LLM provider abstraction.
    Implements thought-action-observation loop for answering circuit design questions.
    Supports Gemini and Claude transparently.
    """
    
    def __init__(self, system: str = ""):
        self.system = system
        self.messages = []
        self.total_cost = 0
        self.total_latency = 0
        self.conversation_history = []  # Provider-agnostic history
        
        # Get the LLM provider
        self.provider = get_llm_provider()
        self.chat_history = []  # Provider-specific history
    
    def __call__(self, message: str) -> str:
        """Send a message to the agent and get a response."""
        result, cost_info = self.execute(message)
        self.total_cost += cost_info.get('total_cost', 0)
        self.total_latency += cost_info.get('latency', 0)
        return result
    
    def execute(self, message: str) -> Tuple[str, Dict]:
        """Execute a single turn of conversation via the LLM provider."""
        start_time = time.time()
        
        try:
            response_text, self.chat_history = self.provider.generate_with_history(
                history=self.chat_history,
                new_message=message,
                system_instruction=self.system,
                temperature=TEMPERATURE,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
            
            # Handle None/empty response
            if not response_text:
                response_text = "Error: Model returned empty response"
            
            end_time = time.time()
            latency = end_time - start_time
            
            # Estimate token counts
            input_tokens = int(len(message.split()) * 1.3) if message else 0
            output_tokens = int(len(response_text.split()) * 1.3) if response_text else 0
            
            cost_info = calculate_cost(input_tokens, output_tokens)
            cost_info['latency'] = latency
            
            return response_text, cost_info
            
        except Exception as e:
            print(f"Error in LLM execution: {e}")
            return f"Error: {str(e)}", {'latency': time.time() - start_time, 'total_cost': 0}
    
    def reset(self):
        """Reset the conversation history."""
        self.conversation_history = []
        self.chat_history = []


# Available actions for the agent
known_actions = {
    "search_db": search_db,
    "load_titles": load_titles
}

# Pattern to match agent actions
action_re = re.compile(r'^Action: (\w+): (.*)$')


def create_image_mapping(observation: Any) -> Dict[str, str]:
    """Create a mapping between figure references and actual image paths."""
    image_map = {}
    
    try:
        if isinstance(observation, str):
            results = json.loads(observation)
        else:
            results = observation
        
        # Extract images and create mapping
        for result in results.get('text', []):
            if result.get('content_type') == 'image':
                path = result['item'].get('path')
                if path and os.path.exists(path):
                    filename = os.path.basename(path)
                    base_name = os.path.splitext(filename)[0]
                    
                    # Extract number from filename (e.g., '21' from 'image_21')
                    if match := re.search(r'image_(\d+)', base_name):
                        img_num = match.group(1)
                        image_map[f"Figure {img_num}"] = filename
                        
    except Exception as e:
        print(f"Error creating image mapping: {e}")
    
    return image_map


def extract_and_save_answer_images(answer_text: str, image_mappings: Dict[str, str], 
                                   observation: Dict) -> Optional[str]:
    """Extract and save images mentioned in the answer."""
    try:
        output_dir = "output_images"
        
        # Clean up existing output directory
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        
        # Find all figure references in the answer
        figure_pattern = r'Figure (\d+)'
        references = re.finditer(figure_pattern, answer_text)
        saved_images = []
        
        for ref in references:
            fig_num = ref.group(1)
            fig_ref = f"Figure {fig_num}"
            
            if fig_ref in image_mappings:
                filename = image_mappings[fig_ref]
                
                # Find the image in the results
                for result in observation.get('text', []):
                    if (result.get('content_type') == 'image' and 
                        'item' in result and 
                        os.path.basename(result['item'].get('path', '')) == filename):
                        
                        source_path = result['item']['path']
                        dest_path = os.path.join(output_dir, filename)
                        
                        if os.path.exists(source_path):
                            shutil.copy2(source_path, dest_path)
                            saved_images.append((fig_ref, dest_path))
                        break
        
        if saved_images:
            print(f"\nSaved {len(saved_images)} images to {output_dir}:")
            for ref, path in saved_images:
                print(f"  - {ref} -> {path}")
            return output_dir
        
        return None
        
    except Exception as e:
        print(f"Error saving images: {e}")
        return None


def query(question: str, max_turns: int = 10) -> Dict[str, Any]:
    """
    Main query function implementing the ReAct agent loop.
    
    Args:
        question: The user's question about analog circuits
        max_turns: Maximum number of agent turns
        
    Returns:
        Dictionary with answer and image paths
    """
    i = 0
    bot = GeminiAgent(system_message)
    next_prompt = question
    image_mappings = {}
    last_observation = None
    final_answer = None
    
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}\n")
    
    while i < max_turns:
        i += 1
        print(f"\n--- Turn {i} ---")
        
        result = bot(next_prompt)
        final_answer = result
        print(result)
        
        # Check if this is the final answer (no more actions)
        if "Action:" not in result:
            image_dir = None
            if last_observation:
                image_dir = extract_and_save_answer_images(result, image_mappings, last_observation)
            
            print(f"\n{'='*60}")
            print(f"Final Answer (Turn {i})")
            print(f"Total estimated cost: ${bot.total_cost:.6f}")
            print(f"Total latency: {bot.total_latency:.2f}s")
            print(f"{'='*60}")
            
            return {
                "answer": final_answer,
                "image_paths": image_dir,
                "total_cost": bot.total_cost,
                "total_latency": bot.total_latency
            }
        
        # Parse and execute actions
        actions = [
            action_re.match(a) 
            for a in result.split('\n') 
            if action_re.match(a)
        ]
        
        if actions:
            action, action_input = actions[0].groups()
            
            if action not in known_actions:
                print(f"Unknown action: {action}")
                next_prompt = f"Observation: Unknown action '{action}'. Available actions are: {list(known_actions.keys())}"
                continue
            
            print(f"\n  Executing: {action}({action_input})")
            
            try:
                # Execute the action
                if action == "load_titles":
                    observation = known_actions[action]()
                else:
                    observation = known_actions[action](action_input)
                
                # Store observation for image extraction
                if isinstance(observation, dict):
                    last_observation = observation
                    # Create image mappings
                    image_mappings.update(create_image_mapping(observation))
                
                # Format observation for next prompt
                if isinstance(observation, dict):
                    obs_str = json.dumps(observation, indent=2, default=str)[:10000]  # Limit size
                else:
                    obs_str = str(observation)[:10000]
                
                next_prompt = f"Observation: {obs_str}"
                
                # Add image mapping info if available
                if image_mappings:
                    next_prompt += f"\n\nImage mappings: {json.dumps(image_mappings)}"
                    
            except Exception as e:
                print(f"Error executing action: {e}")
                next_prompt = f"Observation: Error executing {action}: {str(e)}"
    
    print(f"\nMax turns ({max_turns}) reached")
    return {
        "answer": final_answer or "Could not complete the query within the allowed turns.",
        "image_paths": None,
        "total_cost": bot.total_cost,
        "total_latency": bot.total_latency
    }


def interactive_mode():
    """Run the agent in interactive mode."""
    print("\n" + "="*60)
    print("AnuRAG Interactive Mode")
    print("Type 'quit' or 'exit' to stop")
    print("="*60 + "\n")
    
    while True:
        try:
            question = input("\nYour question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                print("Please enter a question.")
                continue
            
            result = query(question)
            
            print("\n" + "-"*40)
            print("ANSWER:")
            print("-"*40)
            print(result['answer'])
            
            if result.get('image_paths'):
                print(f"\nImages saved to: {result['image_paths']}")
                
        except KeyboardInterrupt:
            print("\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='AnuRAG Agent')
    parser.add_argument('--query', type=str, help='Single query to execute')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if args.query:
        result = query(args.query)
        print("\n" + "="*60)
        print("FINAL ANSWER:")
        print("="*60)
        print(result['answer'])
    elif args.interactive:
        interactive_mode()
    else:
        # Default: interactive mode
        interactive_mode()
