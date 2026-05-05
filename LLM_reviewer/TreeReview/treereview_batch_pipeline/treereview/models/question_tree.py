import uuid

class QuestionNode:
    def __init__(
        self,
        question: str,
        depth: int,
        parent=None,
        children: list = None,
        answer: str = None,
        is_follow_up: bool = False
    ):
        self.uuid = str(uuid.uuid4())
        self.question = question
        self.depth = depth
        self.parent = parent
        self.children = children or []  # list[QuestionNode]
        self.answer = answer
        self.answer_list = []  # list[str]
        self.is_follow_up = is_follow_up
        self._processed = False

    def is_leaf(self):
        return len(self.children) == 0

    def to_dict(self):
        return {
            "uuid": self.uuid,
            "question": self.question,
            "depth": self.depth,
            "children": [child.to_dict() for child in self.children],
            "answer": self.answer,
            "answer_list": self.answer_list,
            "is_follow_up": self.is_follow_up,
            "processed": self._processed,
            "parent_uuid": self.parent.uuid if self.parent else None
        }

    @classmethod
    def from_dict(cls, data):
        node_uuid = data["uuid"]

        node = cls(
            question=data["question"],
            depth=data["depth"],
            answer=data["answer"],
            is_follow_up=data["is_follow_up"]
        )
        node.uuid = node_uuid
        node.answer = data.get("answer")
        node.answer_list = data.get("answer_list")
        node._processed = data.get("processed", False)

        for child_data in data["children"]:
            child = cls.from_dict(child_data)
            child.parent = node
            node.children.append(child)

        return node

    def is_processed(self) -> bool:
        return self._processed

    def mark_processed(self):
        self._processed = True




