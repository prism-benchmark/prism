import json
import os
import time
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from treereview.models.paper import Paper
from treereview.models.question_tree import QuestionNode

if TYPE_CHECKING:  # pragma: no cover
    from treereview.agents.answer_synthesizer import AnswerSynthesizer
    from treereview.agents.question_generator import QuestionGenerator
    from treereview.utility.context_ranker import ContextRanker


class PipelineConfig:
    def __init__(self, max_depth: int = 4, retrieval_top_k: int = 3):
        self.max_depth = max_depth
        self.max_width = {depth: max(0, 6 - depth) for depth in range(self.max_depth) if depth > 0}
        self.max_width[0] = 1
        self.max_expand_width = self.max_width.copy()
        for depth in self.max_expand_width:
            if depth > 1:
                self.max_expand_width[depth] += 2
        self.retrieval_top_k = retrieval_top_k


class ReviewPipeline:
    def __init__(
        self,
        paper: Paper,
        question_generator: "QuestionGenerator",
        context_ranker: "ContextRanker",
        answer_synthesizer: "AnswerSynthesizer",
        config: PipelineConfig = PipelineConfig(),
        state_file: Optional[str] = None,
    ):
        self.paper = paper
        self.question_generator = question_generator
        self.context_ranker = context_ranker
        self.answer_synthesizer = answer_synthesizer
        self.config = config
        self.state_file = state_file or f"checkpoint_{paper.id}.json"
        self.root_node: Optional[QuestionNode] = None
        self.current_stage: Optional[str] = None
        self.build_stack: List[QuestionNode] = []
        self.answer_stack: List[Tuple[QuestionNode, bool]] = []

    def run(self):
        if self._should_resume():
            self.load_checkpoint()
        else:
            self._initialize_pipeline()
        while self.current_stage != "done":
            if self.current_stage == "building_tree":
                self._build_tree_phase()
            elif self.current_stage == "answering":
                self._answer_phase()
            self.save_checkpoint()
        return {
            "full_review": self.root_node.answer if self.root_node else None,
            "feedback_comments": self.root_node.answer_list if self.root_node else [],
        }

    def _initialize_pipeline(self):
        self.root_node = QuestionNode(question=self._get_root_question(), depth=0)
        self.build_stack = [self.root_node]
        self.current_stage = "building_tree"

    def _build_tree_phase(self):
        while self.build_stack:
            current_node = self.build_stack.pop()
            if current_node.depth < self.config.max_depth - 1:
                self._expand_node(current_node)
            self.save_checkpoint()
        self.current_stage = "answering"

    def _expand_node(self, node: QuestionNode):
        max_question = self.config.max_width[node.depth + 1] - len(node.children)
        child_questions = self.question_generator.generate(
            parent_question=node.question,
            paper=self.paper,
            n_questions=max_question,
            node_depth=node.depth,
            max_depth=self.config.max_depth - 1,
        )[:max_question]
        for question in child_questions:
            child_node = QuestionNode(question=question, depth=node.depth + 1, parent=node)
            node.children.append(child_node)
            if child_node.depth < self.config.max_depth - 1:
                self.build_stack.append(child_node)
        self.save_checkpoint()

    def _answer_phase(self):
        if not self.root_node:
            self.current_stage = "done"
            return
        if not self.answer_stack and not self.root_node.is_processed():
            self.answer_stack = [(self.root_node, False)]

        def make_context(chunks: List[str]) -> str:
            return "\n\n".join(chunks)

        while self.answer_stack:
            current_node, visited = self.answer_stack.pop()
            if visited:
                if current_node.is_leaf():
                    chunks = self.context_ranker.rank_chunks(
                        chunks=self.paper.chunks,
                        question=current_node.question,
                        top_k=self.config.retrieval_top_k,
                    )
                    answer = self.answer_synthesizer.summarize(
                        node_type="leaf",
                        question=current_node.question,
                        context=make_context(chunks),
                    )
                    current_node.answer = answer
                    current_node.mark_processed()
                elif current_node.depth > 0:
                    remaining_questions = self.config.max_expand_width[current_node.depth + 1] - len(current_node.children)
                    if remaining_questions < 0:
                        remaining_questions = 0
                    result = self.answer_synthesizer.summarize(
                        node_type="non_leaf",
                        question=current_node.question,
                        children_nodes=current_node.children,
                        max_questions=remaining_questions,
                    )
                    if isinstance(result, list):
                        for question in result:
                            new_node = QuestionNode(
                                question=question,
                                depth=current_node.depth + 1,
                                parent=current_node,
                                is_follow_up=True,
                            )
                            current_node.children.append(new_node)
                            if new_node.depth < self.config.max_depth - 1:
                                self.build_stack.append(new_node)
                        self.answer_stack.append((current_node, False))
                        self.current_stage = "building_tree"
                        self.save_checkpoint()
                        return
                    else:
                        current_node.answer = result
                        current_node.mark_processed()
                self.save_checkpoint()
            else:
                self.answer_stack.append((current_node, True))
                for child in reversed(current_node.children):
                    if not child.is_processed():
                        self.answer_stack.append((child, False))

        if not self.answer_stack:
            self._finalize_review()

    def _finalize_review(self):
        if self.root_node and self.root_node.answer is None and not self.root_node.is_processed():
            self.root_node.answer, self.root_node.answer_list = self.answer_synthesizer.summarize(
                node_type="root",
                children_nodes=self.root_node.children,
                paper=self.paper,
            )
            self.root_node.mark_processed()
            self.current_stage = "done"
        self.save_checkpoint()

    def save_checkpoint(self):
        state = {
            "paper_id": self.paper.id,
            "current_stage": self.current_stage,
            "root_node": self.root_node.to_dict() if self.root_node else None,
            "build_stack_ids": [node.uuid for node in self.build_stack],
            "answer_stack_ids": [[node.uuid, visited] for node, visited in self.answer_stack],
            "timestamp": time.time(),
        }
        temp_file = f"{self.state_file}.tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        os.replace(temp_file, self.state_file)

    def _index_tree(self, node: Optional[QuestionNode]) -> Dict[str, QuestionNode]:
        index: Dict[str, QuestionNode] = {}
        if node is None:
            return index
        stack = [node]
        while stack:
            cur = stack.pop()
            index[cur.uuid] = cur
            stack.extend(cur.children)
        return index

    def load_checkpoint(self):
        with open(self.state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
        if state.get("paper_id") != self.paper.id:
            raise ValueError("Paper ID mismatch")
        self.root_node = QuestionNode.from_dict(state["root_node"]) if state.get("root_node") else None
        node_index = self._index_tree(self.root_node)
        self.build_stack = [node_index[node_id] for node_id in state.get("build_stack_ids", []) if node_id in node_index]
        self.answer_stack = [
            (node_index[node_id], visited)
            for node_id, visited in state.get("answer_stack_ids", [])
            if node_id in node_index
        ]
        self.current_stage = state.get("current_stage")

    def _should_resume(self) -> bool:
        return os.path.exists(self.state_file)

    def _get_root_question(self) -> str:
        return (
            "Generate a comprehensive peer review focusing primarily on identifying and evaluating the weaknesses "
            "of the paper, while also assessing its strengths, scientific contribution, methodological soundness, "
            "and clarity of presentation."
        )
