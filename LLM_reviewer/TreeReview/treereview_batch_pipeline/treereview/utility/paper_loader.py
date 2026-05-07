import re

from treereview.models.paper import Paper
from treereview.utility.text_chunker import TextChunker

class PaperLoader:
    def __init__(self, paper_id: str,
                 paper_path: str,
                 metadata: dict=None,
                 with_appendix: bool=True,
                 chunk_config: dict=None):
        self.paper_id = paper_id
        self.paper_path = paper_path
        self.metadata = metadata
        self.with_appendix = with_appendix
        self.chunk_config = chunk_config or {}
        self.id_to_papers = {}
        self._load()

    def get_paper(self) -> Paper:
        return self._load()

    def _load(self):
        def load_file(path):
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()

        def get_title(content):
            if self.metadata and 'title' in self.metadata:
                return self.metadata.get('title')
            lines = content.split('\n')
            title = ""
            title_started = False
            for line in lines:
                if line.startswith('# '):
                    title_started = True
                    title = line.strip('# ').strip()
                    continue
                if title_started and ("Anonymous authors" in line or line.startswith("#")):
                    break
                if title_started:
                    title += line.strip() + " "
            return title

        def get_abstract(content):
            if self.metadata and 'abstract' in self.metadata:
                return self.metadata.get('abstract')
            lines = content.split('\n')
            abstract = ""
            abstract_started = False
            for line in lines:
                if re.match(r"#+\s+Abstract", line):
                    abstract_started = True
                    continue
                if abstract_started and line.strip().startswith('#'):
                    break
                if abstract_started:
                    abstract += line.strip() + " "
            return abstract.strip()

        def get_toc(content):
            toc = []
            lines = content.split('\n')
            for line in lines:
                if line.startswith('## '):
                    toc.append(line.strip())
                elif line.startswith('### '):
                    toc.append(line.strip())
                elif line.startswith('#### '):
                    toc.append(line.strip())
            return '\n'.join(toc)

        def get_appendix(content):
            appendix_start = re.search(r'#+\s*Appendix', content)
            if not appendix_start:
                return ""
            appendix_text = content[appendix_start.start():]
            return appendix_text

        chunker = TextChunker()
        content = load_file(self.paper_path)
        if self.with_appendix:
            appendix_content = "\n\n" + get_appendix(content)
        else:
            appendix_content = ""
        main_content = content
        sections_to_remove = ['References', 'Acknowledgments', 'Appendix']
        for section in sections_to_remove:
            main_content = re.split(f'(^|\n)#+ *{section}', main_content)[0]
        main_content += appendix_content
        chunks = chunker.chunk(main_content, **self.chunk_config)
        paper = Paper(
            id=self.paper_id,
            title=get_title(content),
            abstract=get_abstract(content),
            toc=get_toc(content),
            chunks=chunks,
            content=main_content,
        )
        return paper


