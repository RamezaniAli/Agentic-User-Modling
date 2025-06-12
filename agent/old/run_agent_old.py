


from langchain_core.messages import HumanMessage
import os
from agent.graph_agent import build_graph, AgentState
from agent.react_agent import build_react_graph, ReactState
from agent.reflect_agent import build_reflect_graph, ReflectState

import json
import re

react_graph = build_react_graph()
reflect_graph = build_reflect_graph()


def run_agent(input_json: dict) -> dict:

    current_dir = os.path.dirname(__file__)
    prompt_path = os.path.join(current_dir, "prompts", "react_human_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as file:
        react_template = file.read()

    current_dir = os.path.dirname(__file__)
    prompt_path = os.path.join(current_dir, "prompts", "reflect_human_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as file:
        reflect_template = file.read()

    filled_react_prompt = react_template.format(
        user_id=input_json["user_id"],
        user_information=input_json["user_information"],
        item_id=input_json["item_id"],
        item_information=input_json["item_information"]
    )
    first_react_msg = HumanMessage(content=filled_react_prompt)

    react_state: ReactState = {
        "user_id": input_json.get("user_id"),
        "user_information": input_json.get("user_information", {}),
        "react_messages": [first_react_msg],
    }

    react_result = react_graph.invoke(react_state) # type: ignore


    filled_reflect_prompt = reflect_template.format(
        user_id=input_json["user_id"],
        user_information=input_json["user_information"],
        item_id=input_json["item_id"],
        item_information=input_json["item_information"],
        true_rating=input_json["true_rating"],
        true_review=input_json["true_review"],
        persona=react_result["persona"],
        predicted_rating=react_result['predicted_rating'],
        predicted_review=react_result['predicted_review'],
        retrieved_interactions=react_result['retrieved_interactions'])
    first_reflect_message = HumanMessage(content=filled_reflect_prompt)

    reflect_state: ReflectState = {
        "reflect_messages": [first_reflect_message],
    }
    reflect_result = reflect_graph.invoke(reflect_state) # type: ignore

    print("++"*50)
    print("==========================================REACT MESSAGES==========================================")
    print("++"*50)

    for msg in react_result["react_messages"]:
        msg.pretty_print()

    print("++"*50)
    print("==========================================Reflect MESSAGES==========================================")
    print("++"*50)

    for msg in reflect_result["reflect_messages"]:
        msg.pretty_print()
    
    result = {
        'user_id':input_json["user_id"],
        'user_information':input_json["user_information"],
        'item_id':input_json["item_id"],
        'item_information':input_json["item_information"],
        'true_rating':input_json["true_rating"],
        'true_review':input_json["true_review"],
        'persona':react_result["persona"],
        'predicted_rating':react_result['predicted_rating'],
        'predicted_review':react_result['predicted_review'],
        'retrieved_interactions':react_result['retrieved_interactions'],
        "updated_interaction": reflect_result['updated_interaction'],
        "updated_retrieved_interactions":reflect_result['updated_retrieved_interactions'],
        "updated_persona":reflect_result['updated_persona'],

    }


    return result
