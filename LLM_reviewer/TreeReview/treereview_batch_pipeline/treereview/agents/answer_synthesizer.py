from typing import List, Optional, Union, Any

from langchain_core.prompts import ChatPromptTemplate

from treereview.models.paper import Paper
from treereview.models.question_tree import QuestionNode
from treereview.utility.utils import load_json_object, load_json_array
from treereview.prompts.templates import *

class AnswerSynthesizer:
    def __init__(self, llm):
        self.llm = llm
        self.leaf_template = ChatPromptTemplate.from_template(LEAF_QUESTION_ANSWER_PROMPT_TEMPLATE)
        self.node_template = ChatPromptTemplate.from_template(ANSWER_AGGREGATION_PROMPT_TEMPLATE)
        self.node_with_max_questions_template = ChatPromptTemplate.from_template(INTERMEDIATE_QUESTION_ANSWER_PROMPT_TEMPLATE)
        self.root_para_template = ChatPromptTemplate.from_template(ROOT_FULL_REVIEW_PROMPT_TEMPLATE)
        self.root_list_template = ChatPromptTemplate.from_template(ROOT_FEEDBACK_COMMENTS_PROMPT_TEMPLATE)

    def summarize(
            self,
            node_type: str = "root", # root, non_leaf, leaf
            question: str = None,
            children_nodes: list[QuestionNode] = None,
            paper: Paper = None,
            context: Optional[str] = None,
            max_questions: Optional[int] = None,

    ) -> Union[str, tuple[Any, Any], tuple[str, Union[list[Any], Any]]]:
        if node_type == "leaf":
            return self._answer_leaf_question(question, context)
        elif node_type == "non_leaf":
            return self._aggregate_answers(question, children_nodes, max_questions)
        elif node_type == "root":
            return self._generate_review(children_nodes, paper)
        else:
            raise ValueError(f"Invalid type: {node_type}")

    def _answer_leaf_question(self, question: str, context: str) -> str:
        messages = self.leaf_template.format_messages(
            question=question,
            context=context
        )
        response = self.llm.invoke(messages)
        return self._extract_content(response)

    def _aggregate_answers(self, question: str, children_nodes: List[QuestionNode], max_questions: int) -> Union[
        tuple[Any, Any], str]:
        questions_answers = [
            f"Sub-questions:{child_node.question}:\nAnswers: {child_node.answer}"
            for child_node in children_nodes
        ]
        questions_answers = "\n\n".join(questions_answers)
        if max_questions and max_questions > 0:
            messages = self.node_template.format_messages(
                question=question,
                questions_answers=questions_answers,
                max_questions=max_questions
            )

            response = self.llm.invoke(messages)
            result_data = load_json_object(response)
            if "synthesized_answer" in result_data:
                answer = result_data["synthesized_answer"]
                cot = result_data["chain_of_thought"]
                return answer
            else:
                question_list = result_data.get("follow_up_questions", "")
                cot = result_data.get("chain_of_thought", "")
                return question_list

        else:
            messages = self.node_with_max_questions_template.format_messages(
                question=question,
                questions_answers=questions_answers
            )
            response = self.llm.invoke(messages)
            result_data = load_json_object(response)
            answer = result_data["synthesized_answer"]
            cot = result_data["chain_of_thought"]
            return answer


    def _generate_review(self, children_nodes: List[QuestionNode], paper: Paper) -> tuple[
        str, Union[list[Any], Any]]:
        questions_answers = [
            f"Sub-questions:{child_node.question}:\nAnswers: {child_node.answer}\n"
            for child_node in children_nodes
        ]
        questions_answers = "\n".join(questions_answers)
        messages = self.root_para_template.format_messages(
            paper_content=paper.content,
            questions_answers=questions_answers
        )
        response = self.llm.invoke(messages)
        review_para = self._extract_content(response)

        messages = self.root_list_template.format_messages(
            paper_content=paper.content,
            questions_answers=questions_answers
        )
        response = self.llm.invoke(messages)
        review_list = load_json_array(response)

        return review_para, review_list

    def _extract_content(self, response: Union[str, dict]) -> str:
        if isinstance(response, str):
            return response.strip()
        elif hasattr(response, "content"):
            return response.content.strip()
        elif "text" in response:
            return response["text"].strip()
        else:
            return str(response).strip()

