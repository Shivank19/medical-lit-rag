import gradio as gr
from rag import rag_setup, query

model, collection, groq_client = rag_setup()

def respond(question):
    if not question.strip():
        return "Please enter a question.", ""
    
    result = query(question, model, collection, groq_client)
    answer = result["answer"]

    sources = "## Sources\n\n"
    for i, source in enumerate(result["sources"], start=1):
        score_pct = int(source["score"] * 100)
        sources += (
            f"**[{i}] {source['title']}**  \n"
            f"📅 {source['year']} &nbsp;|&nbsp; "
            f"📰 {source['journal']} &nbsp;|&nbsp; "
            f"🎯 Relevance: {score_pct}%  \n"
            f"🔗 [View on PubMed]({source['url']})  \n\n"
            f"---\n\n"
        )
    return answer, sources

with gr.Blocks(
    title="Medical Literature QA",
    theme=gr.themes.Soft(),
    js="() => { document.documentElement.setAttribute('data-theme', 'light') }"

) as demo:

    # header
    gr.Markdown(
        """
        <div class="header">

        # 🧬 Medical Literature QA
        ### Answers grounded in peer-reviewed PubMed abstracts, where every claim is cited from over 200 mRNA cancer vaccine papers

        </div>
        """
    )

    # what you can ask
    with gr.Accordion("ℹ️ What can you ask?", open=False):
        gr.Markdown(
            """
            This system answers questions grounded in recent peer-reviewed literature on **mRNA cancer vaccines 
            and immunotherapy**. It works best for:

            - **Mechanism questions**: how mRNA vaccines activate the immune system, how lipid nanoparticles work, 
            how neoantigens are selected
            - **Clinical evidence questions**: what cancer types are being trialled, what trial phases exist, 
            what response rates have been observed
            - **Comparison questions**: how mRNA vaccines compare to traditional immunotherapy, modified vs 
            unmodified mRNA, different delivery platforms
            - **Challenges and limitations**: manufacturing hurdles, tumour heterogeneity, immune escape, 
            regulatory complexity
            - **Specific cancer types**: breast cancer, pancreatic cancer, glioma, prostate cancer, 
            hepatocellular carcinoma, genitourinary cancers
            """
        )
    gr.Markdown(
            """
            > ⚠️ This tool is for **research purposes only** and does not constitute medical advice. 
            Always consult a qualified clinician for medical decisions.
            """
    )

    gr.Markdown("---")

    # input row
    with gr.Row():
        with gr.Column(scale=4):
            question_input = gr.Textbox(
                label="Ask a clinical or research question",
                placeholder="e.g. How do mRNA cancer vaccines activate the immune system?",
                lines=3,
                elem_classes=["question-box"]
            )
        with gr.Column(scale=1, min_width=120):
            submit_btn = gr.Button(
                "Search →",
                variant="primary",
                size="lg"
            )

    # example questions
    gr.Examples(
        examples=[
            ["How do mRNA cancer vaccines activate the immune system?"],
            ["What cancer types are mRNA vaccines currently being trialled for?"],
            ["What are lipid nanoparticles and why are they used in mRNA vaccines?"],
            ["What are the manufacturing challenges for personalised mRNA cancer vaccines?"],
            ["How does anti-PD-1 therapy combine with mRNA vaccines?"],
            ["What is the role of neoantigens in personalised cancer vaccines?"],
            ["How do mRNA vaccines affect the tumour microenvironment?"],
        ],
        inputs=question_input,
        label="Example questions (click any to try)"
    )

    gr.Markdown("---")

    # answer panel: full width
    gr.Markdown("## Answer")
    answer_output = gr.Markdown(
        value="*Your answer will appear here after submitting a question...*",
        elem_id="answer-panel"
    )

    # sources panel: full width below answer
    sources_output = gr.Markdown(
        value="*Retrieved sources will appear here...*",
        elem_id="sources-panel"
    )

    # footer
    with gr.Accordion("ℹ️ How it works", open=False):
        gr.Markdown(
            """
            ---
            Your question is embedded as a vector → ChromaDB finds the 5 most semantically similar abstracts from 200 PubMed papers → Llama 3 reads those abstracts and synthesises a grounded answer citing each source → Every claim links back to a real, verifiable paper.
            """
        )

    # wire interactions
    submit_btn.click(
        fn=respond,
        inputs=[question_input],
        outputs=[answer_output, sources_output]
    )

    question_input.submit(
        fn=respond,
        inputs=[question_input],
        outputs=[answer_output, sources_output]
    )

if __name__ == "__main__":
    demo.launch()