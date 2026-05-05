
class Paper:
    def __init__(self, id: str, title: str, abstract: str, toc: str, chunks: list[str], content: str):
        self.id = id
        self.title = title
        self.abstract = abstract
        self.toc = toc
        self.chunks = chunks
        self.content = content

