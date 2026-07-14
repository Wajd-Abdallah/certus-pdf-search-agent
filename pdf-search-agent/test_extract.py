import re

CITATION_PATTERN = re.compile(r'\[([^\]:]+):p?(\d+)\]')

def extract_citations(answer_text):
    citations = []
    for match in CITATION_PATTERN.finditer(answer_text):
        citations.append({"document": match.group(1).strip(), "page_number": int(match.group(2))})
    return citations

answer1 = "There are four students working on this project. [01_Angebot.pdf:15]"
answer2 = """The main deliverables of this project are:

* A functional PDF search agent [01_Angebot.pdf:15]
* The complete source code [01_Angebot.pdf:15]
* A project documentation [01_Angebot.pdf:15]
* An evaluation of the system [01_Angebot.pdf:15]
* A short user guide [01_Angebot.pdf:15]"""

print(extract_citations(answer1))
print(extract_citations(answer2))
