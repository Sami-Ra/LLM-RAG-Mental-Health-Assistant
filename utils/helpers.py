def format_source_title(filename: str) -> str:
    """
    Convert a PDF filename into a human-readable document title.

    Example
    -------
    anxiety_self_help_guide.pdf
        -> Anxiety Self Help Guide
    """
    return (
        filename.replace("_", " ")
        .replace(".pdf", "")
        .title()
    )


def build_context(retrieved_docs: list[dict]) -> str:
    """
    Build a structured context string from retrieved document chunks.

    The returned context is passed directly to the LLM during
    Retrieval-Augmented Generation (RAG).
    """

    sections = []

    for i, doc in enumerate(retrieved_docs, start=1):

        title = format_source_title(doc["source"])

        sections.append(
            f"""
Source {i}

Document: {title}
Page: {doc['page']}

Content:
{doc['text']}
""".strip()
        )

    return "\n\n" + ("\n\n" + "=" * 50 + "\n\n").join(sections)