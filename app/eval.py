import os
import datasets
from pydantic import SecretStr
from ragas import evaluate
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from rag import rag_setup, query
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

load_dotenv()

# llama 8b for experimentation as it uses less tokens and has more available tokens overall via groq, but can switch to 70b for final evaluation
groq_llm = ChatGroq(
    # model="llama-3.3-70b-versatile"
    model="llama-3.1-8b-instant",
    # api_key=SecretStr(os.getenv('GROQ_API_KEY'))
    api_key=SecretStr(os.getenv("GROQ_API_KEY") or "")
)

evaluator_llm = LangchainLLMWrapper(groq_llm)

embedder = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

def rag_eval(test_cases, model, collection, groq_client):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    results = []

    for i, case in enumerate(test_cases):
        print(f"Evaluating {i+1}/{len(test_cases)}...")
        response = query(case['question'], model, collection, groq_client)
        answer = response['answer']
        contexts = [s['text'] for s in response['sources']]

        # answer relevancy: does the answer address the question?
        q_emb = embedder.encode([case['question']])
        a_emb = embedder.encode([answer])
        relevancy = float(cosine_similarity(q_emb, a_emb)[0][0])

        # context relevance: did retriever find relevant chunks?
        c_embs = embedder.encode(contexts)
        ctx_scores = cosine_similarity(q_emb, c_embs)[0]
        context_score = float(np.mean(ctx_scores))

        # groundedness: does answer overlap with ground truth?
        gt_emb = embedder.encode([case['ground_truth']])
        groundedness = float(cosine_similarity(a_emb, gt_emb)[0][0])

        results.append({
            "question": case['question'][:80],
            "answer_relevancy": round(relevancy, 3),
            "context_relevance": round(context_score, 3),
            "groundedness": round(groundedness, 3)
        })

    avg_relevancy   = np.mean([r['answer_relevancy']  for r in results])
    avg_context     = np.mean([r['context_relevance'] for r in results])
    avg_groundedness = np.mean([r['groundedness']     for r in results])

    summary = {
        "total_questions": len(results),
        "avg_answer_relevancy":   round(float(avg_relevancy), 3),
        "avg_context_relevance":  round(float(avg_context), 3),
        "avg_groundedness":       round(float(avg_groundedness), 3)
    }

    print("\n=== EVALUATION RESULTS ===")
    print(f"Questions evaluated:    {summary['total_questions']}")
    print(f"Answer relevancy:       {summary['avg_answer_relevancy']}")
    print(f"Context relevance:      {summary['avg_context_relevance']}")
    print(f"Groundedness:           {summary['avg_groundedness']}")

    # save to disk
    with open('evaluation/scores.json', 'w') as f:
        json.dump({"summary": summary, "per_question": results}, f, indent=2)
    print("\nSaved to evaluation/scores.json")

    return summary

def rag_eval_RAGAS(test_cases, model, collection, groq_client):
    dataset = []
    for case in test_cases:
        response = query(case['question'], model, collection, groq_client)
        ans = response['answer']
        sources = response['sources']

        dataset.append({
            "question": case['question'],
            "answer": ans,
            "contexts": [s['text'] for s in sources],
            "ground_truth": case['ground_truth']
        })

    eval_dataset = datasets.Dataset.from_list(dataset)

    scores = evaluate(eval_dataset, metrics=[faithfulness, answer_relevancy, context_recall], llm=evaluator_llm, embeddings=embedder)
    return scores

if __name__ == "__main__":
    model, collection, groq_client = rag_setup()
    test_cases = [
        {
            "question": "What makes personalized mRNA cancer vaccines promising for cancer treatment?",
            "ground_truth": "Personalized mRNA cancer vaccines are promising because they can be designed around tumor antigens that activate the patient's immune system against cancer cells. The abstract also says they have a favorable safety profile and are adaptable, which may help improve treatment efficacy and prolong life."
        },
        {
            "question": "What was evaluated in the study of sequential heterogeneous bivalent SARS-CoV-2 vaccinations in long-term care residents?",
            "ground_truth": "The study evaluated the immunogenicity of a sequential vaccination schedule using bivalent Ancestral/Omicron BA.1 and Ancestral/Omicron BA.4/5 mRNA vaccines. It focused on older adults living in long-term care and retirement home settings in Ontario, Canada."
        },
        {
            "question": "What advantages of mRNA therapeutics are highlighted in the review on drug delivery and biomedical applications?",
            "ground_truth": "The review describes mRNA therapeutics as programmable, nonintegrating treatments for infectious disease, cancer, and hereditary disorders. It highlights nucleoside modifications and lipid nanoparticles as key advances that improve stability, cellular entry, and overall translational potential."
        },
        {
            "question": "How can mRNA vaccines affect breast cancer tumors according to the review on current development and future prospects?",
            "ground_truth": "mRNA vaccines can encode tumor-associated antigens, enhance antigen presentation, and activate T cell responses in breast cancer. The review says they may also convert immunologically cold tumors into immune-active ones by reshaping the tumor immune microenvironment."
        },
        {
            "question": "What delivery platforms and future directions are discussed in the review of mRNA vaccine platforms and novel delivery systems?",
            "ground_truth": "The review discusses mRNA engineering strategies and delivery platforms such as lipid nanoparticles, polymeric nanoparticles, virus-like particles, and needle-free administration methods. It also notes ongoing challenges in stability, delivery efficiency, manufacturing, and accessibility, while pointing to artificial intelligence, nanotechnology, and systems immunology as future directions."
        },
        {
            "question": "What therapeutic effect did the sequential selective organ-to-cell targeting strategy achieve in the glioma mRNA vaccine study?",
            "ground_truth": "The strategy enabled selective mRNA expression first in the spleen and then in dendritic cells within the spleen. In mice, the mRNA vaccine treated glioma effectively, especially when combined with anti-PD-1 therapy, and in some cases eradicated the tumors."
        },
        {
            "question": "What did the scoping review find about trends in non-viral mRNA cancer vaccines from 2015 to 2025?",
            "ground_truth": "The review found a shift from ex vivo dendritic cell vaccines to in vivo lipid-based delivery, especially after 2021. It also reported a rise in combination immunotherapy approaches, particularly with immune checkpoint inhibitors, and identified reporting and standardization gaps."
        },
        {
            "question": "What was the goal of the ex vivo mRNA-based immunotherapy platform for endometrial cancer?",
            "ground_truth": "The platform aimed to develop a safe ex vivo mRNA-based vaccine for endometrial cancer. The abstract frames this as a response to the limited effectiveness of current therapies, especially in advanced or chemoresistant disease."
        },
        {
            "question": "What main challenges do cancer vaccines face according to the review of advances, hurdles, and future directions?",
            "ground_truth": "The review says cancer vaccines work by targeting tumor-specific antigens to activate the adaptive immune system. It also notes major obstacles such as tumor heterogeneity, manufacturing scalability, biomarker development, and regulatory complexity."
        },
        {
            "question": "Why are aberrant splicing events important for off-the-shelf mRNA neoantigen vaccines in hepatocellular carcinoma?",
            "ground_truth": "Aberrant splicing events were found more frequently than somatic mutations and produced more immunogenic peptides with broader patient coverage. The study says these splicing-derived neoantigens may be a promising source for off-the-shelf mRNA vaccines in hepatocellular carcinoma."
        },
        {
            "question": "How did active cancer treatment affect antibody responses to a COVID-19 mRNA vaccine?",
            "ground_truth": "Patients undergoing active systemic cancer treatment had significantly lower antibody responses than those not on treatment, especially around the second dose. The study suggests treatment impaired both the quantity and quality of the antibody response and that vaccination timing may need to be tailored."
        },
        {
            "question": "What did the in silico triple-negative breast cancer vaccine study design?",
            "ground_truth": "The study designed both protein and mRNA multi-epitope vaccines targeting triple-negative breast cancer-associated antigens. It selected epitopes from several extracellular and intracellular proteins and reported strong predicted immune stimulation in silico."
        },
        {
            "question": "What is the biological rationale for mRNA vaccines in pancreatic ductal adenocarcinoma?",
            "ground_truth": "The review says mRNA vaccines can deliver patient-specific neoantigens that elicit potent cytotoxic T cell responses against tumor-restricted epitopes. It presents them as a promising way to improve immunogenicity in a disease limited by heterogeneity, late presentation, and an immunosuppressive tumor microenvironment."
        },
        {
            "question": "Why did the nanoparticle adjuvant study focus on the S2 subunit of SARS-CoV-2 spike protein?",
            "ground_truth": "The study focused on the conserved S2 subunit because antibodies against it could provide broader and more durable protection than responses focused on the variable receptor-binding domain. The nanoparticle adjuvant helped overcome immunodominance and promote stronger S2-specific antibody and memory responses."
        },
        {
            "question": "What did the mental model interventions try to correct about mRNA vaccination misconceptions?",
            "ground_truth": "The interventions aimed to reduce unwarranted fears that residual DNA from mRNA vaccine manufacturing could integrate into recipients' DNA and raise cancer or heritable risks. Both preemptive and rebuttal mental-model explanations were effective, and some benefits persisted two months later."
        },
        {
            "question": "What did the macroporous hydrogel-based mRNA cancer vaccine enable in situ?",
            "ground_truth": "The hydrogel-based vaccine enabled in situ recruitment of dendritic cells to the injection site. Those dendritic cells could then take up the mRNA lipoplexes, process the encoded neoantigens, and present them before migrating to draining lymph nodes."
        },
        {
            "question": "What main barriers and design considerations are discussed in the review on enhancing mRNA cancer vaccines?",
            "ground_truth": "The review says the main barriers are low immunogenicity and limited therapeutic efficacy. It focuses on structure engineering, chemical modification, delivery materials, targeted delivery, immune cell interactions, and innate immune stimulation as ways to improve performance."
        },
        {
            "question": "What was the purpose of the systematic review of mRNA vaccines for infectious diseases other than COVID-19?",
            "ground_truth": "The review aimed to summarize the clinical trial evidence for mRNA vaccines targeting infectious diseases other than COVID-19. It was intended to provide an up-to-date evidence base to guide future research and development."
        },
        {
            "question": "Did concurrent asymptomatic Plasmodium falciparum parasitemia reduce neutralizing antibody responses after an mRNA COVID-19 booster?",
            "ground_truth": "No, the study found that concurrent asymptomatic parasitemia did not diminish the neutralizing antibody response after boosting. Both parasitemic and nonparasitemic HIV-infected adults had very high ID50 geometric mean titers one month after vaccination."
        },
        {
            "question": "What are the main themes of the review on DNA and mRNA vaccines in cancer immunotherapy?",
            "ground_truth": "The review focuses on recent progress, stability, in vivo distribution, and delivery challenges for DNA and mRNA vaccines. It also discusses how these platforms can support personalized cancer therapy through improved formulation and delivery technologies."
        },
        {
            "question": "What did the study on pre-vaccination immune profiles try to predict after mRNA vaccination?",
            "ground_truth": "The study examined whether immune profiles before vaccination and responsiveness to innate stimuli could predict reactogenicity and antibody magnitude after mRNA vaccination. It aimed to identify early innate immune correlates that might forecast vaccine responses."
        },
        {
            "question": "What challenges are highlighted in the review on mRNA vaccine development against bacteria?",
            "ground_truth": "The review says bacterial vaccines face greater biological complexity than viral vaccines, which makes antigen selection and mRNA construct design harder. It also highlights translational challenges and notes that bacterial mRNA vaccines are only in early clinical development."
        },
        {
            "question": "How did the first vaccination after hematopoietic stem cell transplantation behave in patients with prior antigen exposure?",
            "ground_truth": "Even with pre-HSCT antigen exposure, the first vaccination after HSCT induced a primary immune response rather than a secondary one. The response appeared later than a true memory response, which suggests prior immunity had not been preserved."
        },
        {
            "question": "What role did the cGAS agonist play in the combined cancer immunotherapy study?",
            "ground_truth": "The cGAS agonist was co-delivered with peptide or mRNA vaccines in lipid nanoparticles to induce type I interferon responses. This improved antigen presentation and potentiated CD8 T cell responses, supporting stronger combination immunotherapy."
        },
        {
            "question": "What happened when MHC-I- and MHC-II-restricted neoantigens were co-administered in the rapid-turnaround mRNA vaccine platform?",
            "ground_truth": "Co-administration increased antigen-specific T cell responses and produced strong anti-cancer efficacy. The platform specifically induced antigen-specific CD8 T cell responses and was designed as a rapid mRNA-based personalized cancer vaccine approach."
        },
        {
            "question": "How did the 5T4 and CD70 mRNA-LNP combination help in prostate cancer therapy?",
            "ground_truth": "The 5T4 antigen provided the tumor target, while CD70 was used to enhance T cell activation and strengthen the immune response. The study explored this combination as a way to improve the potency of mRNA vaccine therapy against prostate cancer."
        },
        {
            "question": "Which orthopoxviruses were targeted by the computational multi-epitope mRNA vaccine design?",
            "ground_truth": "The vaccine was designed against variola virus, vaccinia virus, monkeypox virus, and cowpox virus. The study selected conserved antigen regions from these orthopoxviruses to build a vaccine intended to provoke both cellular and humoral immunity."
        },
        {
            "question": "How did membrane expression affect viral capsid proteins in the mRNA vaccine study?",
            "ground_truth": "Membrane-bound and secreted capsid proteins folded into their native multimeric structures without the viral chaperone, preserving conformational epitopes. They also triggered stronger antibody and T cell responses than the intracellular versions."
        },
        {
            "question": "How did modified and unmodified mRNA differ in the rhesus macaque vaccination study?",
            "ground_truth": "Both modified and unmodified mRNA induced strong but transient innate immune activation after vaccination. Despite differences in cytokine patterns and gene expression, they produced similar levels and kinetics of antigen-specific antibody and T cell responses."
        },
        {
            "question": "What progress and remaining challenges are identified in the influenza vaccine roadmap review?",
            "ground_truth": "The review reports that a minority of roadmap milestones were completed or partially completed, while most were still in progress. Remaining challenges include durable broad protection, immune imprinting, mucosal immunity, correlates of protection, and licensure strategies for broadly protective vaccines."
        }
    ]

    scores = rag_eval(test_cases, model, collection, groq_client)
    print(scores)
    