from langchain_core.prompts import ChatPromptTemplate

from treereview.models.paper import Paper
from treereview.prompts.templates import QUESTION_GENERATOR_PROMPT_TEMPLATE
from treereview.utility.utils import load_json_array


class QuestionGenerator:
    def __init__(self, llm):
        self.prompt = ChatPromptTemplate.from_template(QUESTION_GENERATOR_PROMPT_TEMPLATE)
        self.chain = self.prompt | llm

    def generate(self, parent_question: str, paper: Paper, n_questions: int, node_depth: int, max_depth:int) -> list[str]:
        response = self.chain.invoke({
            "parent_question": parent_question,
            "paper_title": paper.title,
            "paper_abstract": paper.abstract,
            "paper_toc": paper.toc,
            "n_questions": n_questions,
            "node_depth": node_depth,
            "max_depth": max_depth
        })
        return self._parse_response(response)[:n_questions]

    def _parse_response(self, response: str) -> list[str]:
        return load_json_array(response)


