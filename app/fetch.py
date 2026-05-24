import requests
import xml.etree.ElementTree as ET
import json

def search_pubmed(query, max_results):
    url = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={query}&retmax={max_results}&retmode=json'
    response_json = requests.get(url).json()

    response = response_json['esearchresult']
    pmid_list = response['idlist']
    print(f"Found {response['count']} results. Fetching abstracts for PMIDs: {pmid_list}")
    return pmid_list
    
def fetch_abstracts(pmid_list):
    # build a comma separated string of pmids
    pmids = ','.join(pmid_list)
    print(pmids)

    # fetch and return xml
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmids}&rettype=abstract&retmode=xml"
    response_xml = requests.get(url).text
    # print(response_xml)
    return(response_xml)

def parse_xml(xml):
    root = ET.fromstring(xml)
    print(f"Root tag: {root.tag}")

    articles = root.findall('.//PubmedArticle')

    # print(f"Found {len(articles)} articles in XML")

    # # inspect the first article's abstract element raw
    # first = articles[0]
    # abstract_el = first.find('.//AbstractText')
    # print(f"Abstract element: {abstract_el}")
    # print(f"Abstract text: {abstract_el.text if abstract_el is not None else 'ELEMENT NOT FOUND'}")
    
    docs = []
    for article in articles:
        abstract = article.findtext('.//AbstractText')
        if not abstract:
            continue
        pmid = article.findtext('.//PMID')
        docs.append({
            "pmid": pmid,
            "title": article.findtext('.//ArticleTitle'),
            "abstract": abstract,
            "year": article.findtext('.//PubDate/Year'),
            "journal": article.findtext('.//Journal/Title'),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        })
    # print(f"Parsed {len(docs)} articles")
    # print(docs[0])
    return docs


def save_to_json(docs, path):
    with open(path, 'w') as f:
        json.dump(docs, f, indent=4)

if __name__ == "__main__":
    query = '("mRNA+vaccine"+OR+"mRNA-based+vaccine"+OR+"mRNA+immunotherapy")+AND+(cancer+OR+tumor+OR+neoantigen)+AND+("clinical+trial"+OR+"phase+I"+OR+"phase+II"+OR+immunogenicity)+AND+(2019:2025[pdat])'

    max_results = 250

    pmid_list = search_pubmed(query, max_results)
    xml = fetch_abstracts(pmid_list)
    docs = parse_xml(xml)
    path = 'data/abstracts.json'
    save_to_json(docs, path)